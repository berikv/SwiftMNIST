import SwiftUI

struct BrainEngineTraingingView: View {
    let goToValidation: () -> ()
    @State private var index = 0

    let dataset = MNISTDataset.training
    var sample: MNISTSample { dataset[index] }

    var body: some View {
        VStack {
            Text("Train yourself with these input/outputs")
            
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
                        Text("\(number)")
                    }
                }
                GridRow {
                    Text("target")
                    ForEach(0..<10) { number in
                        Text("\(sample.label == number ? "100" : "0")")
                    }
                }
            }

            Rectangle().frame(height: 1)
            HStack {
                Button("Next") {
                    index += 1
                }.disabled(index + 1 == dataset.endIndex)
                
                Button("Validate") {
                    goToValidation()
                }
            }

            Spacer()
        }
    }
}

extension Int: Identifiable {
    public var id: Int {
        self
    }
}

#Preview {
    BrainEngineTraingingView(goToValidation: {})
}
