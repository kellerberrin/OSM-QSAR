"""
Tests for binding pocket detection. 
"""

from __future__ import absolute_import, division, print_function, unicode_literals


__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"


import sys
import mdtraj as md
import tempfile
import os
import shutil
import deepchem as dc
from subprocess import call
from deepchem.feat import hydrogenate_and_compute_partial_charges
from deepchem.dock.binding_pocket import ConvexHullPocketFinder
from deepchem.utils import rdkit_util


from six import with_metaclass

import numpy as np

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMRegression import OSMRegression  # Display and save regression results.
from OSMClassify import OSMClassification  # Display and save classification results.
from OSMModelData import OSMModelData



######################################################################################################
#
# Optional member functions.
#
######################################################################################################


class OSMClassificationTemplate(with_metaclass(ModelMetaClass, OSMClassification)):  # Edit this and change the class name
    # This is a classifier so inherit "with_metaclass(ModelMetaClass, OSMClassification)".

    def __init__(self, args, log):
        super(OSMClassificationTemplate, self).__init__(args, log)  # Edit this and change the class name.

    # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [None], "TYPE": OSMModelData.CLASSES }
                  , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048_5", "SHAPE": None, "TYPE": None } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Test Sub To Include Vina Binding"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "vbind"

    def model_description(self):
        return ("Test Sub To Include Vina Binding")

    def model_define(self): # Should return a model.

        return TestBindingPocket()


    def model_train(self):

        self.model.test_convex_init()
        self.model.test_get_all_boxes(self.args.postfixDirectory)
        self.model.test_boxes_to_atoms(self.args.postfixDirectory)
        self.model.test_convex_find_pockets(self.args.postfixDirectory)
        self.model.test_extract_active_site(self.args.postfixDirectory)

        pose_generator = VinaPoseGenerator()

        protein_file = os.path.join(self.args.postfixDirectory, "PfATP4.pdb")
        ligand_file = os.path.join(self.args.postfixDirectory, "SJ733.pdb")

        pose_generator.generate_poses(protein_file, ligand_file,out_dir=self.args.postfixDirectory)



    def model_read(self): return None  # (Optional) read a model from disk and returns it.

    def model_write(self): pass  # (Optional) write a model to disk.

    def model_prediction(self, data): # prediction and actual are returned as one hot vectors.
        return {"prediction": data.target_data(), "actual": data.target_data() }

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        probability = np.zeros((data.target_data().shape[0], len(self.model_enumerate_classes())), dtype=float)
        probability[:, 0] = 1.0
        return {"probability": probability}

#
# The test rig.
#


def get_molecule_centroid(molecule_xyz):
  """Uses compute centroid and range of 3D coordinents"""
  return np.mean(molecule_xyz, axis=0)


def get_molecule_range(molecule_xyz):
  protein_max = np.max(molecule_xyz, axis=0)
  protein_min = np.min(molecule_xyz, axis=0)
  protein_range = protein_max - protein_min
  return protein_range

class PoseGenerator(object):
  """Abstract superclass for all pose-generation routines."""

  def generate_poses(self, protein_file, ligand_file, out_dir=None):
    """Generates the docked complex and outputs files for docked complex."""
    raise NotImplementedError


def write_conf(receptor_filename,
               ligand_filename,
               centroid,
               box_dims,
               conf_filename,
               exhaustiveness=None):
  """Writes Vina configuration file to disk."""
  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % receptor_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("center_x = %f\n" % centroid[0])
    f.write("center_y = %f\n" % centroid[1])
    f.write("center_z = %f\n\n" % centroid[2])

    f.write("size_x = %f\n" % box_dims[0])
    f.write("size_y = %f\n" % box_dims[1])
    f.write("size_z = %f\n\n" % box_dims[2])

    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)


class VinaPoseGenerator(PoseGenerator):
  """Uses Autodock Vina to generate binding poses."""

  def __init__(self, exhaustiveness=50, detect_pockets=True):
    """Initializes Vina Pose generation"""
    self.exhaustiveness = exhaustiveness
    self.detect_pockets = detect_pockets
    if self.detect_pockets:
      self.pocket_finder = ConvexHullPocketFinder()


  def generate_poses(self,
                     protein_file,
                     ligand_file,
                     centroid=None,
                     box_dims=None,
                     dry_run=False,
                     out_dir=None):
    """Generates the docked complex and outputs files for docked complex."""

    # Prepare receptor
    receptor_name = os.path.basename(protein_file).split(".")[0]
    protein_hyd = os.path.join(out_dir, "%s.pdb" % receptor_name)
    protein_pdbqt = os.path.join(out_dir, "%s.pdbqt" % receptor_name)
    hydrogenate_and_compute_partial_charges(
        protein_file,
        "pdb",
        hyd_output=protein_hyd,
        pdbqt_output=protein_pdbqt,
        protein=True)
    # Get protein centroid and range
    # TODO(rbharath): Need to add some way to identify binding pocket, or this is
    # going to be extremely slow!
    if centroid is not None and box_dims is not None:
      protein_centroid = centroid
    else:
      if not self.detect_pockets:
        receptor_mol = rdkit_util.load_molecule(
            protein_hyd, calc_charges=False, add_hydrogens=False)
        protein_centroid = get_molecule_centroid(receptor_mol[0])
        protein_range = get_molecule_range(receptor_mol[0])
        box_dims = protein_range + 5.0
      else:
        print("About to find putative binding pockets")
        pockets, pocket_atoms_maps, pocket_coords = self.pocket_finder.find_pockets(
            protein_file, ligand_file)
        # TODO(rbharath): Handle multiple pockets instead of arbitrarily selecting
        # first pocket.
        print("Computing centroid and size of proposed pocket.")
        pocket_coord = pocket_coords[0]
        protein_centroid = np.mean(pocket_coord, axis=1)
        pocket = pockets[0]
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = pocket
        x_box = (x_max - x_min) / 2.
        y_box = (y_max - y_min) / 2.
        z_box = (z_max - z_min) / 2.
        box_dims = (x_box, y_box, z_box)

    # Prepare receptor
    ligand_name = os.path.basename(ligand_file).split(".")[0]
    ligand_hyd = os.path.join(out_dir, "%s.pdb" % ligand_name)
    ligand_pdbqt = os.path.join(out_dir, "%s.pdbqt" % ligand_name)

    # TODO(rbharath): Generalize this so can support mol2 files as well.
    hydrogenate_and_compute_partial_charges(
        ligand_file,
        "sdf",
        hyd_output=ligand_hyd,
        pdbqt_output=ligand_pdbqt,
        protein=False)
    # Write Vina conf file
    conf_file = os.path.join(out_dir, "conf.txt")
    write_conf(
        protein_pdbqt,
        ligand_pdbqt,
        protein_centroid,
        box_dims,
        conf_file,
        exhaustiveness=self.exhaustiveness)

    # Define locations of log and output files
    log_file = os.path.join(out_dir, "%s_log.txt" % ligand_name)
    out_pdbqt = os.path.join(out_dir, "%s_docked.pdbqt" % ligand_name)
    # TODO(rbharath): Let user specify the number of poses required.
    print("About to call Vina")
    call("vina --config %s --log %s --out %s" % (conf_file, log_file, out_pdbqt), shell=True)
    # TODO(rbharath): Convert the output pdbqt to a pdb file.

    # Return docked files
    return protein_hyd, out_pdbqt




#
# The test rig.
#

class TestBindingPocket(object):
  """
  Does sanity checks on binding pocket generation. 
  """

  def test_convex_init(self):
    """Tests that ConvexHullPocketFinder can be initialized."""
    print("Convex_Init")
    finder = dc.dock.ConvexHullPocketFinder()

  def test_get_all_boxes(self, postfix_directory):
    """Tests that binding pockets are detected."""
    print("Test_All_Boxes")
    protein_file = os.path.join(postfix_directory, "PfATP4.pdb")
    ligand_file = os.path.join(postfix_directory, "SJ733.pdb")
    coords = rdkit_util.load_molecule(protein_file)[0]

    boxes = dc.dock.binding_pocket.get_all_boxes(coords)
    assert isinstance(boxes, list)
    # Pocket is of form ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    for pocket in boxes:
      assert len(pocket) == 3
      assert len(pocket[0]) == 2
      assert len(pocket[1]) == 2
      assert len(pocket[2]) == 2
      (x_min, x_max), (y_min, y_max), (z_min, z_max) = pocket
      assert x_min < x_max
      assert y_min < y_max
      assert z_min < z_max

  def test_boxes_to_atoms(self, postfix_directory):
    """Test that mapping of protein atoms to boxes is meaningful."""
    protein_file = os.path.join(postfix_directory, "PfATP4.pdb")
    ligand_file = os.path.join(postfix_directory, "SJ733.pdb")
    coords = rdkit_util.load_molecule(protein_file)[0]
    boxes = dc.dock.binding_pocket.get_all_boxes(coords)

    mapping = dc.dock.binding_pocket.boxes_to_atoms(coords, boxes)
    assert isinstance(mapping, dict)
    for box, box_atoms in mapping.items():
      (x_min, x_max), (y_min, y_max), (z_min, z_max) = box
      for atom_ind in box_atoms:
        atom = coords[atom_ind]
        assert x_min <= atom[0] and atom[0] <= x_max
        assert y_min <= atom[1] and atom[1] <= y_max
        assert z_min <= atom[2] and atom[2] <= z_max

  def test_compute_overlap(self):
    """Tests that overlap between boxes is computed correctly."""
    # box1 contained in box2
    box1 = ((1, 2), (1, 2), (1, 2))
    box2 = ((1, 3), (1, 3), (1, 3))
    mapping = {box1: [1, 2, 3, 4], box2: [1, 2, 3, 4, 5]}
    # box1 in box2, so complete overlap
    np.testing.assert_almost_equal(
        dc.dock.binding_pocket.compute_overlap(mapping, box1, box2), 1)
    # 4/5 atoms in box2 in box1, so 80 % overlap
    np.testing.assert_almost_equal(
        dc.dock.binding_pocket.compute_overlap(mapping, box2, box1), .8)

  def test_merge_overlapping_boxes(self):
    """Tests that overlapping boxes are merged."""
    # box2 contains box1
    box1 = ((1, 2), (1, 2), (1, 2))
    box2 = ((1, 3), (1, 3), (1, 3))
    mapping = {box1: [1, 2, 3, 4], box2: [1, 2, 3, 4, 5]}
    boxes = [box1, box2]
    merged_boxes, _ = dc.dock.binding_pocket.merge_overlapping_boxes(mapping,
                                                                     boxes)
    print("merged_boxes")
    print(merged_boxes)
    assert len(merged_boxes) == 1
    assert merged_boxes[0] == ((1, 3), (1, 3), (1, 3))

    # box1 contains box2
    box1 = ((1, 3), (1, 3), (1, 3))
    box2 = ((1, 2), (1, 2), (1, 2))
    mapping = {box1: [1, 2, 3, 4, 5, 6], box2: [1, 2, 3, 4]}
    boxes = [box1, box2]
    merged_boxes, _ = dc.dock.binding_pocket.merge_overlapping_boxes(mapping,
                                                                     boxes)
    print("merged_boxes")
    print(merged_boxes)
    assert len(merged_boxes) == 1
    assert merged_boxes[0] == ((1, 3), (1, 3), (1, 3))

    # box1 contains box2, box3
    box1 = ((1, 3), (1, 3), (1, 3))
    box2 = ((1, 2), (1, 2), (1, 2))
    box3 = ((1, 2.5), (1, 2.5), (1, 2.5))
    mapping = {
        box1: [1, 2, 3, 4, 5, 6],
        box2: [1, 2, 3, 4],
        box3: [1, 2, 3, 4, 5]
    }
    merged_boxes, _ = dc.dock.binding_pocket.merge_overlapping_boxes(mapping,
                                                                     boxes)
    print("merged_boxes")
    print(merged_boxes)
    assert len(merged_boxes) == 1
    assert merged_boxes[0] == ((1, 3), (1, 3), (1, 3))

  def test_convex_find_pockets(self, postfix_directory):
    """Test that some pockets are filtered out."""
    protein_file = os.path.join(postfix_directory, "PfATP4.pdb")
    ligand_file = os.path.join(postfix_directory, "SJ733.pdb")

    protein = md.load(protein_file)

    finder = dc.dock.ConvexHullPocketFinder()
    all_pockets = finder.find_all_pockets(protein_file)
    pockets, pocket_atoms_map, pocket_coords = finder.find_pockets(protein_file,
                                                                   ligand_file)
    # Test that every atom in pocket maps exists
    n_protein_atoms = protein.xyz.shape[1]
    print("protein.xyz.shape")
    print(protein.xyz.shape)
    print("n_protein_atoms")
    print(n_protein_atoms)
    for pocket in pockets:
      pocket_atoms = pocket_atoms_map[pocket]
      for atom in pocket_atoms:
        # Check that the atoms is actually in protein
        assert atom >= 0
        assert atom < n_protein_atoms

    assert len(pockets) < len(all_pockets)


  def test_extract_active_site(self, postfix_directory):
    """Test that computed pockets have strong overlap with true binding pocket."""
    protein_file = os.path.join(postfix_directory, "PfATP4.pdb")
    ligand_file = os.path.join(postfix_directory, "SJ733.pdb")

    active_site_box, active_site_atoms, active_site_coords = (
        dc.dock.binding_pocket.extract_active_site(protein_file, ligand_file))
    finder = dc.dock.ConvexHullPocketFinder()
    pockets, pocket_atoms, _ = finder.find_pockets(protein_file, ligand_file)

    # Add active site to dict
    print("active_site_box")
    print(active_site_box)
    pocket_atoms[active_site_box] = active_site_atoms
    overlapping_pocket = False
    for pocket in pockets:
      print("pocket")
      print(pocket)
      overlap = dc.dock.binding_pocket.compute_overlap(pocket_atoms,
                                                       active_site_box, pocket)
      if overlap > .5:
        overlapping_pocket = True
      print("Overlap for pocket is %f" % overlap)


