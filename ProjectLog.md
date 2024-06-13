
# Project log

In the project log I'll describe my thinking along the way.

## Goal

In this project I'll learn about simple Neural Networks, implementing them
without any mathematical libraries for assistence.
I'll cover different overfitting, input/hidden/output layers, and may get to 
try convolutional layers.

## Invariants

It makes sense to shape this porject by setting boundaries. This project will:

* Not take longer than one month
* Use the Xcode / Swift toolchain, targetting MacOS
* Use no libraries / packages / frameworks except for those already included in Swift
* Solve a known NN problem

## High level operation

The Neural Network will have two different programs associated: Trainging and Evaluation.

In freeform code, the Trainging program looks like:

    for (input, expected) in trainingSet {
        output = evaluate(input, on: bias)
        error = output - expected
        improve(bias, reducing: error)
    }

The Evaluation program then simply becomes:

    output = evaluate(input, on: bias)
    
The `evaluate(, on:)` and `improve(, reducing:)` functions work with `bias`.
Bias is a collection of numbers, grouped in a certain way to make the Neural Network work.
Let's dive a bit deeper on how a Neural Network is build.

The input is some form of data. It can be a single number, or a vector, or a 2d set of numbers, etc.
What's important is that the shape of the input is known and does not change. You can't feed arbitrary
text into a Neural Network, because the text is of an unknown length. There are clever tricks to allow
Neural Networks to work with text input, but that is out of scope of this project.

Instead we'll be use the MNIST dataset for this project. MNIST is a dataset, specifically made to test
the performance of Neural Networks. It consists of sets of handwritten letters.
    
## Log

### 2024-06-10

- Set up the project
- Define project invariants
- Describe high level functionality

## 2024-06-12

- Read and present the MNIST dataset on the app
- Rename project to SwiftMNIST
- Add an appicon

## 2024-06-13

- Add some vector algebra support for Array (will probably move to simd types soon)
- Create a BrainEngine that lets a human perform the same training and evaluation as the Neural Network does
