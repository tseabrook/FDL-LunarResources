import split_and_merge as sm
import sys
sys.setrecursionlimit(10000)

node0 = sm.Node([0,1])
node1 = sm.Node([0,2])
node2 = sm.Node([1,1])
node3 = sm.Node([1,2])

#1 -> 2 -> 3 -> 1

lines = [None] * 4
lines[0] = sm.Line(0,0,0,0,0)
lines[1] = sm.Line(0,0,0,0,0)
lines[2] = sm.Line(0,0,0,0,0)
lines[3] = sm.Line(0,0,0,0,0)

lines[0].nodes[0] = node0
lines[0].nodes[1] = node1
lines[1].nodes[0] = node1
lines[1].nodes[1] = node2
lines[2].nodes[0] = node2
lines[2].nodes[1] = node0
lines[3].nodes[0] = node0
lines[3].nodes[1] = node3

node0.edges.append(lines[0])
node1.edges.append(lines[0])

node1.edges.append(lines[1])
node2.edges.append(lines[1])

node0.edges.append(lines[2])
node2.edges.append(lines[2])

node1.edges.append(lines[3])
node3.edges.append(lines[3])

#path = sm.Path(node1)

cycles = sm.find_nxCycle(lines)
print(path.cycles)