import Foundation
import simd

struct DecimalOutputLayer: LayerType {
    typealias SIMD = SIMD16<Float>

    typealias Input = [SIMD]
    typealias Output = [Float]

    private(set) var weights: [[SIMD16<Float>]] // 10 x (28*28)
    private(set) var bias: [Float] // 10

    static let inputSize = 28*28
    static let outputSize = 10

    init() {
        func random() -> Float { Float.random(in: -0.1 ... 0.1) }
        bias = [Float]((0..<Self.outputSize).map { _ in random() })
        weights = (0..<Self.outputSize).map { _ in
            [SIMD16<Float>](packing: (0..<Self.inputSize).map { _ in random() })
        }
    }

    func forward(input: Input) -> Output {
        //        let result = (0..<10).map { _ in Float.random(in: -1..<1) }

        let result = weights.indices.map { outputIndex in
            let weights = weights[outputIndex]

            let output = input.indices.map { inputIndex in
                let result = input[inputIndex] * weights[inputIndex]
                return simd_reduce_add(result)
            }.sum

            return output + bias[outputIndex]
        }

        assert(result.count == 10)

        // softmax

        // Shift to avoid exp() overflowing
        let shifted = result .- result.max()!
        let expsum = shifted.map { exp($0) }.sum
        let softmax = shifted.map { exp($0) / expsum }

        return softmax
    }

    func backward(input: Input, output: Output, gradient: Output) -> Input {
        var inputGradient = [SIMD](repeating: SIMD(repeating: 0.0), count: Self.inputSize / SIMD.scalarCount)

        for j in 0..<Self.outputSize {
            for i in 0..<Self.inputSize / SIMD.scalarCount {
                inputGradient[i] += gradient[j] * weights[j][i]
            }
        }

        return inputGradient
    }

    func computeGradients(input: Input, output: Output, gradient: Output) -> (weightGradients: [Input], biasGradients: Output) {
        var weightGradients = [[SIMD]](repeating: [SIMD](repeating: SIMD(repeating: 0.0), count: Self.inputSize / SIMD.scalarCount), count: Self.outputSize)

        var biasGradients = [Float](repeating: Float.zero, count: Self.outputSize)

        for i in 0..<weightGradients.count {
            for j in 0..<weightGradients[i].count {
                weightGradients[i][j] = gradient[i] * input[j]
            }
            biasGradients[i] = gradient[i]
        }

        return (weightGradients, biasGradients)
    }

    mutating func updateParameters(weightGradients: [Input], biasGradients: Output, learningRate: Float) {
        for i in 0..<weights.count {
            for j in 0..<weights[i].count {
                weights[i][j] -= learningRate * weightGradients[i][j]
            }
            bias[i] -= learningRate * biasGradients[i]
        }
    }
}
