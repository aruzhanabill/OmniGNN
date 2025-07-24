## OmniGNN: Globalized Approach to Multirelational GNNs

[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

globalizing the topology of graph neural networks

TODO: Fill out this long description.

## Introduction 
In recent years, Graph Neural Networks (GNNs) have been established as the standard for prediction tasks on graph datasets at the graph-, edge-, and node-level, with significant applications in areas such as molecular modeling, transportation networks, and recommendation systems. Particularly, in financial domains, recent fusion models that pair Graph Convolutional Networks (GCNs) with temporal Recurrent Neural Networks (RNNs) like Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs) have set the benchmark for dynamic node-level prediction tasks using structures like stock correlation or company knowledge graphs. These models, however, face several limitations. First, although RNNs are well-suited for capturing long-range temporal dependencies, their sequential nature causes memory bottlenecks with the increase in complexity, limiting their capacity to integrate information across the full temporal history. Second, message-passing GCNs that rely solely on one-hop neighborhood aggregation fail to distinguish nonisomorphic graphs with similar local structures. Additionally, the standard GCN aggregation scheme applies uniform averaging to incoming messages, which fails to acknowledge the varying relevance of such messages. Third, many GNNs face the challenge of oversmoothing, generating homogeneous node embeddings after multiple rounds of message-passing, making it difficult for the GNN to learn longer-term dependencies in the graph. While skip connections are commonly utilized to preserve previous node-level information during the update step, this strategy increases the parameter count and model depth, leaving the graph susceptible to oversmoothing 

![Alt text](OmniGNN.png)

*Fig. 1: OmniGNN Model Architecture.*


## Install

```sh
```

## Usage

```sh
```

## Maintainers
[@amberhli] (https://github.com/amberhli)
[@aruzhanabill](https://github.com/aruzhanabill)


## Contributing
PRs accepted.

Small note: If editing the README, please conform to the
[standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

MIT © 2025 Aruzhan Abil
