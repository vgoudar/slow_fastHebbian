# slow_fastHebbian
Slow network that learns via backdrop, fast network that implements hebbian learning (gated by slow network)

Meta-learning network with a fast biological RNN (gated hebbian plasticity) coupled with a slow biologicial RNN (trained to adaptively perform a task with the help of the fast RNN).
The slow network gates platicity (i.e. weight changes) in the fast network

Network performs 3back task and a WCST analog (see task.py for description of tasks)
WCST analog currently trains on up to 6 rules

Can be used to study:
(1) gating functions of plasticity in neuromodulatory systems
(2) emergent function of networks with fast hebbian component

To run, call train.py <seed>
Select task in train.py by setting task to 'WCST' or '3Back'
