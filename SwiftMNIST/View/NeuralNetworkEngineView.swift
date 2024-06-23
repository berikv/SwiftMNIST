import SwiftUI
import Charts

@MainActor
struct NeuralNetworkEngineView: View {
    let trainer = NeuralNetworkTrainer()

    var body: some View {
        let biggestMSE = trainer.trainingBatchResults.map(\.meanSquaredError).max()
        Chart {
            ForEach(trainer.trainingBatchResults) { trainingBatchResult in
                LineMark(
                    x: .value("index", trainingBatchResult.numberOfSamples),
                    y: .value("error", trainingBatchResult.meanSquaredError / biggestMSE!))
                .foregroundStyle(by: .value("Value", "Training MSE"))
            }

            ForEach(trainer.validationResults) { validationResult in
                BarMark(
                    x: .value("index", validationResult.numberOfSamplesTrained),
                    y: .value("Validation correct", validationResult.correctPct))
                .foregroundStyle(by: .value("Value", "Validation Accuracy"))
                .annotation(position: .topTrailing) {
                    Text("\(validationResult.correctPct * 100, format: .number.precision(.fractionLength(1)))%")
                }
            }
        }
        .chartForegroundStyleScale([
            "Training MSE": .blue,
            "Validation Accuracy": .teal,
        ])

        VStack {
            HStack {
                Button("Train") {
                    trainer.train(epochs: 20)
                }.disabled(trainer.isTraining)

                Button("Validate") {
                    trainer.validate()
                }.disabled(trainer.isValidating)

                Button("Stop") {
                    trainer.stop()
                }.disabled(!trainer.isTraining && !trainer.isValidating)
            }

            VStack {
                Text("Epoch: \(trainer.epoch) learning rate: \(trainer.learningRate)")

                if trainer.isTraining && trainer.isValidating {
                    Text("Training and Validating")
                } else if trainer.isTraining {
                    Text("Training")
                } else if trainer.isValidating {
                    Text("Validating")
                } else {
                    Text("Idle")
                }

                if let trainingStart = trainer.startDate, let trainingStop = trainer.stopDate {
                    let rate = Double(trainer.trainingSampleCount) / trainingStop.timeIntervalSince(trainingStart)
                    Text("Training took \(relative: trainingStart...trainingStop) " + "at \(rate, specifier: "%.0f") samples per second")
                    Text("Processed \(trainer.trainingSampleCount) samples in total")
                } else if let trainingStart = trainer.startDate {
                    Text("Training for \(trainingStart, style: .relative) at \(trainer.trainingRate, specifier: "%.0f") samples per second")
                }

                if !trainer.trainingBatchResults.isEmpty {
                    Text("Error \(trainer.trainingBatchResults.map(\.meanSquaredError).mean())")
                }

                if let validationResults = trainer.validationResults.last {
                    let mseMean = validationResults.meanSquaredError
                    let correctCount = validationResults.correctCount
                    let totalCount = validationResults.numberOfSamples
                    let correctPct = Double(correctCount) / Double(totalCount) * 100
                    Text("Validation mean MSE \(mseMean), correct \(correctPct, specifier: "%.0f")%, \(correctCount)/\(totalCount)")
                }
            }
        }
    }
}

#Preview {
    NeuralNetworkEngineView()
}
