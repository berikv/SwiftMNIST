import Foundation
import simd

let hiddenLayerSize = 256

struct NeuralEngine {

    private let learningRateStart: Float = 0.0005
    private let learningRateFactor: Float = 0.5
    private let L2factor: Float = 0.00001

    private(set) var epoch: Int = 0

    var learningRate : Float {
        learningRateStart * pow(learningRateFactor, Float(epoch))
    }

    private let inputLayer = BitmapInputLayer()
    private var hiddenLayer = HiddenLayer(inputSize: 28*28, outputSize: hiddenLayerSize)
    private var batchNormLayer = BatchNormLayer(size: hiddenLayerSize)
    private var droputLayer = DropoutLayer(rate: 0.03)
    private var outputLayer = DecimalOutputLayer()

    mutating func nextEpoch() {
        epoch += 1
    }

    mutating func train(_ sample: MNISTSample) -> [Float] {

//        Single layer approach:
//        let input = inputLayer.forward(input: sample.image)
//        let output = outputLayer.forward(input: input)
//
//        let lossGradient = zip(output, sample.target).map { $0.0 - $0.1 }
//        let gradients = outputLayer.computeGradients(input: input, gradient: lossGradient)
//
//        outputLayer.updateParameters(weightGradients: gradients.weightGradients, biasGradients: gradients.biasGradients, learningRate: 0.1)
//
//        return output

        // Forward pass
        let inputLayerOutput = inputLayer.forward(input: sample.image)
        let hiddenLayerOutput = hiddenLayer.forward(input: inputLayerOutput)
        try! trapOutOfBounds(hiddenLayerOutput, bounds: -50...50)
        let batchNormLayerOutput = batchNormLayer.forwardTrain(input: hiddenLayerOutput)
        let dropoutLayerOutput = droputLayer.forward(input: batchNormLayerOutput)
        let outputLayerOutput = outputLayer.forward(input: dropoutLayerOutput)

        // Compute the gradient of the loss with respect to the output
        //        Why This Works:
        //
        //            •    Softmax: It converts logits to probabilities. The softmax function ensures that the predicted probabilities sum to 1.
        //            •    Cross-Entropy Loss: It measures the difference between the predicted probabilities and the actual labels. By taking the gradient of this loss with respect to the logits, we get a vector that tells us how to adjust the logits to reduce the loss.
        let lossGradient = zip(outputLayerOutput, sample.target).map { $0.0 - $0.1 }
        try! trapOutOfBounds(lossGradient)

        // Backward pass through the output layer to get the gradient with respect to hidden layer output
        let hiddenLayerGradient = outputLayer.backward(input: hiddenLayerOutput, output: outputLayerOutput, gradient: lossGradient)
        try! trapOutOfBounds(hiddenLayerGradient, bounds: -50...50)

        // Compute gradients and update parameters for the output layer
        let outputGradients = outputLayer.computeGradients(input: hiddenLayerOutput, gradient: lossGradient)
        try! trapOutOfBounds(outputGradients.biasGradients)
        try! trapOutOfBounds(outputGradients.weightGradients, bounds: -50...50)

        // Compute gradients and update parameters for the hidden layer
        let hiddenLayerGradients = hiddenLayer.computeGradients(input: inputLayerOutput, gradient: hiddenLayerGradient)
        try! trapOutOfBounds(hiddenLayerGradients.biasGradients, bounds: -50...50)
        try! trapOutOfBounds(hiddenLayerGradients.weightGradients, bounds: -100...100)

        let outputLayerL2Gradients = outputLayer.weights.map { $0.map { 2 * L2factor * $0 } }
        let hiddenLayerL2Gradients = hiddenLayer.weights.map { $0.map { 2 * L2factor * $0 } }

        let adjustedOutputWeightGradients = zip(outputGradients.weightGradients, outputLayerL2Gradients).map { zip($0, $1).map { $0 + $1 } }
        let adjustedHiddenWeightGradients = zip(hiddenLayerGradients.weightGradients, hiddenLayerL2Gradients).map { zip($0, $1).map { $0 + $1 } }

        // Update the layer parameters
        outputLayer.updateParameters(weightGradients: adjustedOutputWeightGradients, biasGradients: outputGradients.biasGradients, learningRate: learningRate)
        hiddenLayer.updateParameters(weightGradients: adjustedHiddenWeightGradients, biasGradients: hiddenLayerGradients.biasGradients, learningRate: learningRate)

        // Backward pass through the hidden layer is not needed as the input layer does not learn
//        let inputLayerGradient = hiddenLayer.backward(input: inputLayerOutput, output: hiddenLayerOutput, gradient: hiddenLayerGradient)

        return outputLayerOutput
    }

    func evaluate(_ sample: MNISTSample) -> [Float] {
        let input = inputLayer.forward(input: sample.image)
        let hidden = hiddenLayer.forward(input: input)
        let batchNorm = batchNormLayer.forward(input: hidden)
        let output = outputLayer.forward(input: batchNorm)
        return output
    }
}

struct OutOfBoundsError: Error {}
func trapOutOfBounds<F: BinaryFloatingPoint>(_ num: F, bounds: ClosedRange<F> = -1...1) throws {
    if !bounds.contains(num) {
//        throw OutOfBoundsError()
    }
}

func trapOutOfBounds<F: BinaryFloatingPoint>(_ array: [F], bounds: ClosedRange<F> = -1...1) throws {
    for e in array { try trapOutOfBounds(e, bounds: bounds) }
}

func trapOutOfBounds<F: BinaryFloatingPoint>(_ array: [[F]], bounds: ClosedRange<F> = -1...1) throws {
    for a in array {
        try trapOutOfBounds(a, bounds: bounds)
    }
}
