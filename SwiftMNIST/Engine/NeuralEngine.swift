import Foundation

struct NeuralEngine {

    func train(_ sample: MNISTSample) -> [Double] {
        Array(repeating: 0.1, count: 10)
    }

    func evalute(_ sample: MNISTSample) -> [Double] {
        return if Bool.random() {
            Array(repeating: 0.1, count: 10)
        } else {
            sample.target
        }
    }

}
