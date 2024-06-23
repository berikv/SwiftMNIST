import Foundation

struct TrainingBatchResult: Identifiable {
    var id: Int { numberOfSamples }
    let numberOfSamples: Int
    let meanSquaredError: Float
}

struct ValidationResult: Identifiable {
    var id: Int { numberOfSamples }
    let epoch: Int
    let numberOfSamplesTrained: Int
    let numberOfSamples: Int
    let meanSquaredError: Float
    let correctCount: Int
    let errors: [MNISTSample]
    var correctPct: Double { Double(correctCount) / Double(numberOfSamples) }
}

@Observable
@MainActor
final class NeuralNetworkTrainer {
    private(set) var startDate: Date?
    private(set) var stopDate: Date?
    private(set) var trainingSampleCount = 0
    private(set) var trainingBatchResults = [TrainingBatchResult]()
    private(set) var validationResults = [ValidationResult]()
    private(set) var trainingRate: Double = 0

    var epoch: Int { engine.epoch }
    var learningRate: Float { engine.learningRate }

    private var trainTask: Task<(), any Error>?
    private var validateTask: Task<(), any Error>?

    var isTraining: Bool { trainTask != nil }
    var isValidating: Bool { validateTask != nil }

    private var engine = NeuralEngine()

    func train(epochs: Int) {
        guard trainTask == nil else { return }
        
        if engine.epoch == epochs {
            return
        }

        let trainingSet = MNISTDataset.training.shuffled()
        stopDate = nil
        startDate = Date()

        trainTask = Task.detached { [weak self, epochs] in
            var engine = await self!.engine
            var lastUpdate = Date()
            var batch = [Float]()

            for sample in trainingSet {
                let prediction = engine.train(sample)
                let error = meanSquaredError(prediction: prediction, target: sample.target)
                batch.append(error)
                
                let timePassed = Date().timeIntervalSince(lastUpdate)
                if timePassed > 1/10 {
                    try Task.checkCancellation()
                    Task { @MainActor [self, engine, batch] in
                        self?.trainingRate = Double(batch.count) / timePassed
                        self?.updateTraining(engine: engine, errorBatch: batch)
                    }
                    lastUpdate = Date()
                    batch.removeAll(keepingCapacity: true)
                }
            }

            engine.nextEpoch()
            
            await self?.updateTraining(engine: engine, errorBatch: batch)
            await self?.finalizeTraining()

            Task.detached { [self, epochs] in await self?.train(epochs: epochs) }
        }
    }

    func stop() {
        guard let trainTask else { return }
        trainTask.cancel()
        finalizeTraining()
    }

    func validate() {
        let engine = self.engine
        let numberOfSamplesTrained = self.trainingSampleCount

        validateTask = Task.detached { [weak self, engine, numberOfSamplesTrained] in
            var meanSquaredErrors = [Float]()
            var errors = [MNISTSample]()
            var correctCount = 0

            for sample in MNISTDataset.validation {
                let prediction = engine.evaluate(sample)
                let mse = meanSquaredError(prediction: prediction, target: sample.target)
                meanSquaredErrors.append(mse)
                if prediction.firstIndex(of: prediction.max()!) == Int(sample.label) {
                    correctCount += 1
                } else {
                    errors.append(sample)
                }
            }

            Task { @MainActor [self, meanSquaredErrors, correctCount, errors, engine] in
                let result = ValidationResult(
                    epoch: engine.epoch,
                    numberOfSamplesTrained: numberOfSamplesTrained,
                    numberOfSamples: meanSquaredErrors.count,
                    meanSquaredError: meanSquaredErrors.mean(), 
                    correctCount: correctCount,
                    errors: errors)
                self?.validationResults.append(result)
                self?.validateTask = nil
            }
        }
    }

    func reset() {
        trainTask?.cancel()
        trainTask = nil

        validateTask?.cancel()
        validateTask = nil

        startDate = nil
        stopDate = nil

        trainingSampleCount = 0
        trainingBatchResults.removeAll(keepingCapacity: true)
        validationResults.removeAll(keepingCapacity: true)

        engine = NeuralEngine()
    }

    private func updateTraining(engine: NeuralEngine, errorBatch: [Float]) {
        self.engine = engine
        trainingSampleCount += errorBatch.count
        trainingBatchResults.append(TrainingBatchResult(
            numberOfSamples: trainingSampleCount,
            meanSquaredError: errorBatch.mean()))
    }

    private func finalizeTraining() {
        stopDate = Date()
        trainTask = nil

        validate()
    }
}
