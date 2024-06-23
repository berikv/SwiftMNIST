import Foundation

struct DropoutLayer {
    let rate: Float
    init(rate: Float) {
        self.rate = rate
    }

    func forward(input: [Float]) -> [Float] {
        return input.map { Float.random(in: 0...1) < rate ? 0.0 : $0 }
    }
}
