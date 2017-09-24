This project is about finding topologies of artificial neural networks with evolutionary search and training the individual models of every found form with backpropagation techniques.

The topologies are constrained to be made of special layers called "abstractors", that are fully-connected layers with an arbitrary number of input unities (usually 2) and only one output unity - in the future it may be allowed to have as many output unities as necessary, but never more than the number of input unities. The fitness of a topolgy is a function of how well it's models trained with backpropagation generalize to unseen data, their training cost given the data, and their cost on testing data - the time used to reach a predetermined accuracy level may be included into the fitness too.

A model in the path of the evolutionary search is a composition of abstractors and a final fully-connected layer between the last activations from the abstractor composition and the output unities.

