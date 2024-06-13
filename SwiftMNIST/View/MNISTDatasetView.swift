import SwiftUI
import CoreGraphics

struct MNISTDatasetView: View {
    @Binding var dataset: MNISTDataset
    let columns = Array(repeating: GridItem(.flexible(minimum: 25, maximum: 50)), count: 8)
    var body: some View {
        ScrollView(.vertical) {
            LazyVGrid(columns: columns) {
                ForEach(dataset, id: \.self) { sample in
                    Text("\(sample.label)")
                    Image(
                        bitmap: sample.image,
                        width: dataset.width,
                        height: dataset.height)
                }
            }
        }
    }
}

#Preview {
    MNISTDatasetView(dataset: .constant(MNISTDataset.training))
}
