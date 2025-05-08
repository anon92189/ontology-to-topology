# Data and Code for "From Ontology to Topology"

This repository contains all data and code referenced in the paper "From Ontology to Topology: The Societal Context Reference Frame as an Organizing Framework for Sociotechnical Concepts." Specifically, it contains the following three folders:

--classifications: .csv files containing the assignment of at least societal context ontology classs (i.e., agent, artifact, or precept) to each of the classes in the OMRSE, FMO, and SHAS ontologies, along with a justification of each assignment decision.

--rdfs: .rdf files containing the OMRSE ontology, the FMO ontology, and the combined ontology containing OMRSE, FMO, and SHAS.

--python: python scripts that: 1) check the assignments of upper ontology classes for inconsistencies; 2) generate the .rdf implementation of the combined ontology; and 3) generate the custom embedding for each class in our combined ontology, generate Figure 2, and calculate the correlation between cosine similarity of embeddings and minimum pathlength between classes in the knowledge graph represenation of our combined ontology.
