# Class and Method Documentation

Documentation for CAST's classes.

## Operation of a network

A Network object starts empty.

Network components (layers, branching structure, etc.) are added one at a time to the Network object.

Network components are added to branches, which are separate paths of execution through a network.  
Each branch has a unique numerical ID. The original branch in a network has ID 0.

Operators (layers and activation functions) don't change the number of branches.

Adding a Splitter component creates additional branches, which can then be added to.

A Combiner component merges branches into the Combiner's branch. When a branch is combined, the branch's ID is never reused. Attempting to add to a merged branch results in an error.

To train and optimize, a network must have exactly one non-merged branch.

<br>

Example of branch structure creation:
1. A new, empty network is created. Branch 0 becomes available.
2. A two-way Splitter is added to branch 0. The addition of the Splitter creates branch 1.

At this point, the network cannot be used for training and prediction. The network has multiple unterminated branches (0 and 1).

3. Operators are added to both branch 0 and 1.
4. A Combiner is placed into branch 0, set to merge branch 1. Branch 1 ends at the newly added Combiner.

Adding a component to branch 1 is no longer possible because branch 1 has been merged.

5. A two-way Splitter is placed in branch 0. Branch 2 is created (branch 1's ID is not reused).
6. More operators are added to branches 0 and 2. A Combiner merges branches 0 and 2. Now that there is only one unterminated branch, the network can be trained.

---


Classes:
* [ActivationFunction](class_docs/activation_function.md)
* [NetworkComponent](class_docs/network_component.md)
* [Network](class_docs/network.md)
* [Optimizer](class_docs/optimizer.md)
* [Output Stream Manipulators](class_docs/ostream_manip.md)