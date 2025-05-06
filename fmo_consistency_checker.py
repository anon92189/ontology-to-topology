"""
FMO Consistency Checker

Checks the consistency of the upper-ontology classifications of all classes in
the FMO ontology.

"""


#Install rdflib
!pip install rdflib

#Load packages
from rdflib import Graph, RDF, RDFS, OWL
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def get_upper(class_param: str) -> list:
  """
  Obtains the upper ontology classifications for any class in the OMRSE,
  FMO, or SHAS ontologies.

  Args:
    - class_param: the name of the ontology class.

  Returns:
    - A list of upper ontology classifications.
  """
  classifications = list(scrf_df.loc[
      scrf_df['Original Ontology Class'] == class_param, 'SCRF Classification'
  ])
  if len(classifications) == 0:
      return []
  else:
    return [label for label in ['Agent', 'Artifact', 'Precept'] if label in
            classifications[0]]


def get_label(entity):
  """Returns the label for an entity in the RDF."""
  for label in g.objects(entity, RDFS.label):
      return str(label)


# Load RDF file of FMO ontology
g = Graph()
g.parse("/rdfs/FMO.rdf", format="xml")


# Load classifications for the FMO ontology
scrf_df = pd.read_csv('/classifications/FMO_SCRF.csv')


# Collect all declared classes
classes = set()
for cls in g.subjects(RDF.type, OWL.Class):
    classes.add(cls)
for cls in g.subjects(RDF.type, RDFS.Class):
    classes.add(cls)



# Extract rdfs:subClassOf relations between declared classes
subclass_relations = {}
for c in classes:
  if c is not None:
    subclasses = []
    for subclass, _, _ in g.triples((None, RDFS.subClassOf, c)):
        if subclass in classes:
            subclass_label = get_label(subclass)
            subclasses.append(subclass_label)
    subclass_relations[get_label(c)] = subclasses


#Run the consistency check
results = []
for c in subclass_relations.keys():
  if c is not None:
    key_upper = get_upper(c)
    for s in subclass_relations[c]:
      subclass_upper = get_upper(s)
      for u in subclass_upper:
        if u not in key_upper:
          results.append(f'{s} is a {u}, but {c} is not a {u}.')

if len(results) == 0:
  print('No inconsistencies found.')
if len(results) > 0:
  for r in results:
    print(r)
