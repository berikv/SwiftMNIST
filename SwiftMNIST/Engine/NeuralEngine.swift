import Foundation
import simd

struct NeuralEngine {
    let inputLayer = BitmapInputLayer()
    var outputLayer = DecimalOutputLayer()

    mutating func train(_ sample: MNISTSample) -> [Float] {
        let input = inputLayer.forward(input: sample.image)
        let output = outputLayer.forward(input: input)

        let lossGradient = zip(output, sample.target).map { $0.0 - $0.1 }
        let gradients = outputLayer.computeGradients(input: input, output: output, gradient: lossGradient)

        outputLayer.updateParameters(weightGradients: gradients.weightGradients, biasGradients: gradients.biasGradients, learningRate: 0.1)

        return output
    }

    func evaluate(_ sample: MNISTSample) -> [Float] {
        let input = inputLayer.forward(input: sample.image)
        let output = outputLayer.forward(input: input)
        return output
    }

}

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

struct BitmapInputLayer: InputLayerType {
    typealias SIMD = SIMD16<Float>

    func forward(input: Data) -> [SIMD] {
        [SIMD](packing: input.map { Float($0) / Float(UInt8.max) })
    }
}

extension Array where Element == SIMD16<Float> { // SIMDScalar
    typealias SIMD = SIMD16<Float>

    init<C: RandomAccessCollection>(packing input: C) where C.Element == Float {
        var index = input.startIndex

        self.init()
        reserveCapacity(input.count / SIMD.scalarCount)

        while index < input.endIndex {
            // let upperBound = input.index(index, offsetBy: SIMD.scalarCount, limitedBy: input.endIndex)
            var upperBound = index
            guard input.formIndex(&upperBound, offsetBy: SIMD.scalarCount, limitedBy: input.endIndex)
            else { fatalError() }

            //            let paddingCount = max(0, upperBound - input.endIndex)
            let paddingCount = input.distance(from: upperBound, to: input.endIndex) % SIMD.scalarCount

            // Using Array(repeating: Float.zero, ...) makes the Swift compiler go nuts
            // Explicitly typing the Element as Float works around that
            let zeroPadding = [Float](repeating: .zero, count: paddingCount)
            let segmentWithPadding = input[index..<upperBound] + zeroPadding

            let element = SIMD16<Float>(segmentWithPadding)
            append(element)

            index = upperBound
        }
    }
}

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
