import Foundation

protocol InputLayerType {
    associatedtype Input
    associatedtype Output
    func forward(input: Input) -> Output
}

protocol LayerType {
    associatedtype Input
    associatedtype Output

    func forward(input: Input) -> Output
    func backward(input: Input, output: Output, gradient: Output) -> Input
    func computeGradients(input: Input, gradient: Output) -> (weightGradients: [Input], biasGradients: Output)
    mutating func updateParameters(weightGradients: [Input], biasGradients: Output, learningRate: Float)
}
