"""
Combined RDF Generator

Creates a single RDF for all subClassOf relations defined within the OMRSE, FMO, and SHAS ontologies, 
with the SCRF ontology of agents, artifacts, and precepts layered on top of it to create a unified 
ontology of sociotechnical concepts.
"""

#Import Packages
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
import pandas as pd
import numpy as np
import re

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

def sanitize(name):
    """Sanitize class names"""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-]", "", name)  # keep alphanumerics, underscores, hyphens
    if name and name[0].isdigit():
        name = "_" + name  # prepend underscore if it starts with a digit
    return name


# Load RDF files
g = Graph()
g.parse("/rdfs/OMRSE.rdf", format="xml")
g.parse("/rdfs/FMO.rdf", format="xml")

# Collect all declared classes
classes = set()
for cls in g.subjects(RDF.type, OWL.Class):
    classes.add(cls)
for cls in g.subjects(RDF.type, RDFS.Class):
    classes.add(cls)

# Helper to get a label or a readable fallback
def get_label(entity):
    for label in g.objects(entity, RDFS.label):
        return str(label)

# Extract rdfs:subClassOf relations between declared classes in OMRSE and FMO
subclass_relations = {}

for c in classes:
  if c is not None:
    subclasses = []
    for subclass, _, _ in g.triples((None, RDFS.subClassOf, c)):
        if subclass in classes:
            subclass_label = get_label(subclass)
            subclasses.append(subclass_label)
    subclass_relations[get_label(c)] = subclasses


# Manually input the SHAS subClassOf relations
subclass_relations['Representational Harms'] = ['Stereotyping Social Groups',
      'Demeaning Social Groups', 'Erasing Social Groups',
      'Alientating Social Groups',
      'Denying People the Opportunity to Self-Indentify',
       'Reifying Essentialist Social Categories']

subclass_relations['Allocative Harms'] = ['Opportunity Loss','Economic Loss']

subclass_relations['Quality of Service Harms'] = ['Alienation',
      'Increased Labor','Service/Benefit Loss']

subclass_relations['Interpersonal Harms'] = ['Loss of Agency',
      'Tech-facilitated Violence','Diminished Health and Wlel-being',
      'Privacy Violations']

subclass_relations['Social System Harms'] = ['Information Harms',
      'Cultural Harms', 'Civic and Political Harms', 'Socio-economic Harms',
                                             'Environmental Harms']


#Load classifications for all three ontologies and extract the subclass
#relations.
fmo_df = pd.read_csv("/classifications/FMO_SCRF.csv")
omrose_df = pd.read_csv("/classifications/OMRSE_SCRF.csv")
shas_df = pd.read_csv("/classifications/SHAS_SCRF.csv")
scrf_df = pd.concat([fmo_df, omrose_df, shas_df])


subclass_relations['Agent'] = [c for c in subclass_relations.keys() if 'Agent'
                                                  in get_upper(c)]

subclass_relations['Artifact'] = [c for c in subclass_relations.keys() if
                                  'Artifact' in get_upper(c)]

subclass_relations['Precept'] = [c for c in subclass_relations.keys() if
                                  'Precept' in get_upper(c)]


# Create a new RDF graph
g = Graph()

# Define a namespace for the ontology
EX = Namespace("http://example.org/ontology/")
g.bind("ex", EX)

# Collect all unique class names
all_classes = set(subclass_relations.keys())
for subclasses in subclass_relations.values():
    all_classes.update(subclasses)

# Declare each class as an rdfs:Class and add a label
for class_name in all_classes:
  if class_name is not None:
    sanitized_name = sanitize(class_name)
    class_uri = EX[sanitized_name]
    g.add((class_uri, RDF.type, RDFS.Class))
    g.add((class_uri, RDFS.label, Literal(class_name)))

# Add subclass relationships
for parent, subclasses in subclass_relations.items():
  if parent is not None:
    parent_uri = EX[sanitize(parent)]
    for subclass in subclasses:
      if subclass is not None:
        subclass_uri = EX[sanitize(subclass)]
        g.add((subclass_uri, RDFS.subClassOf, parent_uri))

#Save to file
g.serialize(destination="/rdfs/combined_ontology.rdf", format="xml")
