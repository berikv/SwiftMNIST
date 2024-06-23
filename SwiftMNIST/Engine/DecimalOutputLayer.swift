import Foundation
import simd

struct DecimalOutputLayer: LayerType {

    typealias Input = [Float]
    typealias Output = [Float]

    private(set) var weights: [[Float]]
    private(set) var bias: [Float]

    let inputSize = hiddenLayerSize
    let outputSize = 10

    private var inputIndices: Range<Int> { (0..<inputSize) }
    private var outputIndices: Range<Int> { (0..<outputSize) }

    init() {
        func random() -> Float { Float.random(in: -0.1 ... 0.1) }
        bias = [Float](count: outputSize) { random() }
        weights = [[Float]](count: outputSize) { [inputSize] in
            [Float](count: inputSize) { random() }
        }
    }

    func forward(input: Input) -> Output {
        assert(input.count == inputSize)
        
        let result = outputIndices.map { outputIndex in
            (input .* weights[outputIndex]).sum + bias[outputIndex]
        }

        assert(result.count == outputSize)

        // softmax

        // Shift to avoid exp() overflowing
        let shifted = result .- result.max()!
        let expsum = shifted.map { exp($0) }.sum
        let softmax = shifted.map { exp($0) / expsum }

        return softmax
    }

    func backward(input: Input, output: Output, gradient: Output) -> Input {
        var inputGradient = [Float](repeating: 0.0, count: inputSize)

        for outputIndex in outputIndices {
            for inputIndex in inputIndices {
                inputGradient[inputIndex] += gradient[outputIndex] * weights[outputIndex][inputIndex]
            }
        }

        return inputGradient
    }

    func computeGradients(input: Input, gradient: Output) -> (weightGradients: [Input], biasGradients: Output) {
        var weightGradients = [[Float]](repeating: [Float](repeating: 0.0, count: inputSize), count: outputSize)
        let biasGradients = gradient

        for i in 0..<weightGradients.count {
            weightGradients[i] = (input .* gradient[i]).clip(distanceFromZero: 10)
        }

        return (weightGradients, biasGradients)
    }

    mutating func updateParameters(weightGradients: [Input], biasGradients: Output, learningRate: Float) {
        for i in 0..<weights.count {
            weights[i] = (weights[i] .- weightGradients[i] .* learningRate)
            bias = bias .- (biasGradients .* learningRate)
        }
    }
}
