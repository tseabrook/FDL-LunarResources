# Traverse Planning

The initial goal of the FDL2017 Lunar Water and Volatiles group was to develop an improved automated approach to traverse planning, that would automatically search over terrains and illuminations, given a rover specification and mapping data.

Some preliminary work was performed to begin the framework for traverse planning, including the outline for the necessary components.
Scripts within include:
**ConnectedComponents** - Later used for crater detection, this script clusters neighbouring pixels that satisfy boundary threshold conditions.

Groundwork for inclusive of KDTree and A* search algorithms is included, however the ConnectedComponents algorithm highlighted issues with false artefacts in the Digital Elevation Model, which then became the focus of our project.
