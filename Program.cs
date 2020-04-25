using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TinyYolov3
{
    class Program
    {
        static void Main(string[] args)
        {
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;

            using (var session = new InferenceSession(@"D:\Ubuntu\onnx_c#\project\TinyYolov3\TinyYolov3\Assets\yolov3-tiny.onnx", options))
            {
                var input = session.InputMetadata;
                var output = session.OutputMetadata;
                var container = new List<NamedOnnxValue>();
                //var container = new List<NamedOnnxValue>();
                float[] inputData0 = LoadTensorFromFile(@"D:\Ubuntu\onnx_c#\project\TinyYolov3\TinyYolov3\Assets\test_data_set_0\input_0.txt"); //image data
                float[] inputData1 = LoadTensorFromFile(@"D:\Ubuntu\onnx_c#\project\TinyYolov3\TinyYolov3\Assets\test_data_set_0\input_1.txt"); //image shape

                foreach (var name in input.Keys)
                {
                    Console.WriteLine(name);
                }
                foreach (var name in output.Keys)
                {
                    Console.WriteLine(name);
                }
                Console.WriteLine(input.Count);
                Console.WriteLine(output.Count);

                ///// Create Inputs
                var nov = new List<NamedOnnxValue>();
                var tensor0 = new DenseTensor<float>(inputData0, new int[] { 1, 3, 416, 416 });
                //var tensor1 = new DenseTensor<float>(new float[] { 375, 500 }, new int[] { 1, 2 });
                var tensor1 = new DenseTensor<float>(inputData1, new int[] { 1, 2 });

                container.Add(NamedOnnxValue.CreateFromTensor<float>("input_1", tensor0));
                container.Add(NamedOnnxValue.CreateFromTensor<float>("image_shape", tensor1));

                // Run the inference
                using (var results = session.Run(container))  // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    // dump the results
                    foreach (var r in results)
                    {
                        //Console.WriteLine(r.Name);
                        //Console.WriteLine(r.AsTensor<int>().GetArrayString());
                        //Console.WriteLine(r.AsTensor<int>().GetArrayString());
                        if (r.Name == "yolonms_layer_1")
                        {
                            File.WriteAllText(@"D:\Ubuntu\onnx_c#\project\TinyYolov3\TinyYolov3\Assets\test_data_set_0\out_0.txt", r.AsTensor<float>().GetArrayString());
                        }
                        if (r.Name == "yolonms_layer_1:1")
                        {
                            File.WriteAllText(@"D:\Ubuntu\onnx_c#\project\TinyYolov3\TinyYolov3\Assets\test_data_set_0\out_1.txt", r.AsTensor<float>().GetArrayString());
                        }
                        if (r.Name == "yolonms_layer_1:2")
                        {
                            File.WriteAllText(@"D:\Ubuntu\onnx_c#\project\TinyYolov3\TinyYolov3\Assets\test_data_set_0\out_2.txt", r.AsTensor<int>().GetArrayString());
                        }
                    }
                }
            }
            while (true)
            {
            }
        }

        static float[] LoadTensorFromFile(string filename)
        {
            string[] lines = File.ReadAllLines(filename, Encoding.UTF8);
            var tensorData = new List<float>();
            foreach (string line in lines)
            {
                //Console.WriteLine(line);
                tensorData.Add(Single.Parse(line));
            }
            return tensorData.ToArray();
        }
    }
}
