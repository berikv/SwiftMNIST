import SwiftUI

struct BrainEngineValidationView: View {

    let dataset = MNISTDataset.validation
    var sample: MNISTSample { dataset[index] }

    @State private var index = 0
    @State private var errors = [Double]()
    @State private var prediction: [Double] = Array(repeating: 0, count: 10)

    var body: some View {
        VStack {
            Text("Validate yourself with these input/outputs")

            Rectangle().frame(height: 1)
            Text("Input")

            Image(bitmap: sample.image, width: dataset.width, height: dataset.height)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 200)

            Rectangle().frame(height: 1)
            Text("Output")
            Grid {
                GridRow {
                    Text("number")
                    ForEach(0..<10) { number in
                        Text("\(Int(number))")
                    }
                }
                GridRow {
                    Text("probability")
                    ForEach(0..<10) { number in
                        Text("\(Int(prediction.normalized[number] * 100))")
                    }
                }
                GridRow {
                    Button("Clear") {
                        prediction = Array(repeating: 0, count: 10)
                    }
                    ForEach(0..<10) { number in
                        Button("+") {
                            prediction[number] += 1
                        }
                    }
                }
            }

            Rectangle().frame(height: 1)
            HStack {
                Text("Total accured error \(errors.mean)")
                Button("Next") {
                    let error = meanSquaredError(
                        prediction: prediction.normalized,
                        target: sample.target)

                    errors.append(error)

                    prediction = Array(repeating: 0, count: 10)
                    index += 1
                }.disabled(index + 1 == dataset.endIndex)


            }
            
            Button("Start over") {
                prediction = Array(repeating: 0, count: 10)
                index = 0
                errors.removeAll(keepingCapacity: true)
            }

            Spacer()
        }
    }
}

#Preview {
    BrainEngineValidationView()
}


