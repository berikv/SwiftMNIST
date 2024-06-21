import XCTest
@testable import SwiftMNIST

final class PerformanceTests: XCTestCase {

    func testTrainPerformance() throws {
        let trainingSet = MNISTDataset.training.shuffled()[...5000]
        var mseSum = Float.zero

        measure {
            var engine = NeuralEngine()
            for sample in trainingSet {
                let prediction = engine.train(sample)
                mseSum += meanSquaredError(prediction: prediction, target: sample.target)
            }
        }

        XCTAssert(mseSum != 0)
    }

    func testValidatePerformance() {
        let validationSet = MNISTDataset.validation.shuffled()
        var engine = NeuralEngine()

        var mseSum = Float.zero

        measure {
            for sample in validationSet {
                let prediction = engine.evaluate(sample)
                mseSum += meanSquaredError(prediction: prediction, target: sample.target)
            }
        }
        
        // Make sure the engine.evaluate is not optimised away
        XCTAssert(mseSum != 0)
    }

}
