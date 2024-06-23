import Foundation
import simd

extension Array {
    @inline(__always)
    init(count: Int, createElement: () -> Element) {
        self = (0..<count).map { _ in createElement() }
    }
}

struct HiddenLayer: LayerType {
    typealias Input = [Float]
    typealias Output = [Float]
    typealias Gradient = Output
    typealias Weights = [[Float]]
    typealias Bias = [Float]

    private(set) var weights: [[Float]]
    private(set) var bias: [Float]

    private let inputSize: Int// = 28*28
    private let outputSize: Int// = hiddenLayerSize

    private var inputIndices: Range<Int> { (0..<inputSize) }
    private var outputIndices: Range<Int> { (0..<outputSize) }

    init(inputSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.outputSize = outputSize

//        func random() -> Float { Float.random(in: -0.01 ... 0.01) }
        func heRandom(size: Int) -> Float {
            let stddev = (2 / Float(size)).squareRoot()
            return Float.random(in: -1...1) * stddev
        }

        bias = [Float](count: outputSize) { [inputSize] in heRandom(size: inputSize) }
        weights = [[Float]](count: outputSize) { [inputSize] in
            [Float](count: inputSize) { heRandom(size: inputSize) }
        }
    }

    func forward(input: Input) -> Output {
        assert(input.count == inputSize)
        let result = outputIndices.map { outputIndex in
            (input .* weights[outputIndex]).sum + bias[outputIndex]
        }

        assert(result.count == outputSize)

        // ReLu
        return result.map { max(0, $0) }
    }

    func backward(input: Input, output: Output, gradient: Output) -> Input {
        // ReLu gradient: output > 0 ? 1 : 0
        let gradient = zip(output, gradient).map { (output, gradient) in output > 0 ? gradient : 0 }

        var inputGradient = [Float](repeating: 0.0, count: inputSize)

        for outputIndex in outputIndices {
            for inputIndex in inputIndices {
                inputGradient[inputIndex] += gradient[outputIndex] * weights[outputIndex][inputIndex]
            }
        }

        return inputGradient
    }

    func computeGradients(input: Input, gradient: Gradient) -> (weightGradients: Weights, biasGradients: Bias) {
        var weightGradients = [[Float]](repeating: [Float](repeating: 0.0, count: inputSize), count: outputSize)
        let biasGradients = gradient

        for i in 0..<weightGradients.count {
            let gradients = input .* gradient[i]
            let clipped = gradients//.clip(distanceFromZero: 1)
            weightGradients[i] = clipped
        }

        return (weightGradients, biasGradients)
    }

    mutating func updateParameters(weightGradients: Weights, biasGradients: Bias, learningRate: Float) {
        for i in 0..<weights.count {
            weights[i] = (weights[i] .- (weightGradients[i] .* learningRate))//.clip(min: -1, max: 1)
            bias[i] -= learningRate * biasGradients[i]
        }
    }

}
