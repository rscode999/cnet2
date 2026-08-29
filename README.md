# CAST (Compartmentalized Autograd System with Tensors)
*CNet 2.0*  
**C++ framework for neural networks**

## Foreword

I have worked on the CNet project throughout the 2025-26 school year. When I started work on CNet in August 2025, I had no formal education in deep learning, so the CNet project became my de facto deep learning class. I learned about the strengths and weaknesses of different activation functions, how an optimizer works, and the intricacies of the C++ language. I take great pride in the final product, knowing that I built a solid framework and mastered the extremely fine details of deep learning.

But I needed *more power...*

The seeds of a CNet successor arrived at the end of the 2025-26 school year.  
To aid in future image-recognition projects, I wanted to add convolutional layers and the Adam optimizer. I soon learned that CNet was not as expansion-friendly as I thought. The forward pass logic supported only 1D linear layers without branching structures. Worse, the existing SGD optimizer code turned out to be bloated and unmaintainable. I had also learned more about how deep learning frameworks are implemented, explaining the unusual constructs of the PyTorch framework, especially why the optimization pass is not called through the network. Worse, the Eigen package that CNet uses did not support tensors of arbitrary dimension.

I realized that the best way to accomplish my goals and expand CNet is to rewrite the framework from the ground up.

In the summer of 2026, I started work on a CNet successor with these design goals:

- Allow users to create networks of any architecture
- Make all training and validation operations available using only XTensor's `xt::xarray`, with no need to wrap it in another class. Operations and underlying logic become much simpler.
- Decentralize logic and data as much as possible. Avoid a situation where a single class or method takes the majority of the logic, becoming bloated.
    - Reduce the number of fields in the network. Put data and functionality (i.e. gradients of layer weights) within the network fields.
- Use architecture where the network is the central object. All operations can be done through a network.
    - Ideally, CNet's 3 main operations (forward, predict, backward/reverse) are called as network methods

Guided by its design goals, the CAST framework took shape over the coming months.

## Setup Instructions

You will need a compiler that supports C++20 or later.

### XTensor Installation

This project requires the XTensor package and some of its supporting libraries.

Steps:

1. Go to each of these links: [XTensor](https://github.com/xtensor-stack/xtensor), [XTensor BLAS](https://github.com/xtensor-stack/xtensor-blas), [XTL](https://github.com/xtensor-stack/xtl)

2. On each github repo, find the "XXX Tags" page. The Tags button should be near the branch selection dropdown. Ensure that the following required version tags exist:  
- XTensor: 0.27.0  
- XTensor BLAS: 0.23.0  
- XTL: 0.8.2


3. Open a terminal in the "lib" folder. Clone each repo, with the required version tags only:
```
git clone --branch <VERSION TAG> <REPO LINK>

# Example
git clone --branch 0.27.0 https://github.com/xtensor-stack/xtensor
```

You should have 3 directories inside the "lib" folder: xtensor, xtensor-blas, and xtl. Each should contain the "include" subdirectory, which contains the library source files.

### Other Setup

Due to the large size of the XTensor libraries, the C/C++ VS Code extension is not recommended. A better alternative is ClangD, which is much faster and gives better warnings.  
Configuration files come with this repo.

NOTE: If you want to use ClangD, ensure [MSVC Build Tools](https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc?view=msvc-170) is installed.  
Ensure the options for "Desktop development with C/C++" and MSVC Build Tools for x64/x86" are selected.

## Documentation

See the [documentation file](docs/documentation.md).