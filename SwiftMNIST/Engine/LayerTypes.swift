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
//    func backward(output: Output, error: Float)
}
