using Microsoft.ML;
using Microsoft.ML.Data;

// 1. Initalize ML.NET environment
MLContext mlContext = new();

const string path = @"D:\\WorkSpaces\\VsProjects\\ML.NET_Example\\ML.NET_Example\\";

// 2. Load training data
IDataView trainData = mlContext.Data.LoadFromTextFile<ModelInput>(path + "taxi-fare-train.csv", separatorChar: ',');

// 3. Add data transformations
var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: "PaymentTypeEncoded", "PaymentType")
    .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features",
    "PaymentTypeEncoded", "PassengerCount", "TripTime", "TripDistance"));

// 4. Add algorithm
var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "FareAmount", featureColumnName: "Features");

var trainingPipeline = dataProcessPipeline.Append(trainer);

// 5. Train model
var model = trainingPipeline.Fit(trainData);

// 6. Evaluate model on test data
IDataView testData = mlContext.Data.LoadFromTextFile<ModelInput>(path + "taxi-fare-test.csv");
IDataView predictions = model.Transform(testData);
var metrics = mlContext.Regression.Evaluate(predictions, "FareAmount");

// 7. Predict on sample data and print results
var input = new ModelInput
{
    PassengerCount = 1,
    TripTime = 1150,
    TripDistance = 4,
    PaymentType = "CRD"
};

var result = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model).Predict(input);

Console.WriteLine($"Predicted fare: {result.FareAmount}\nModel Quality (RSquared): {metrics.RSquared}");




return;

public class ModelInput
{
    [LoadColumn(2)]
    public float PassengerCount;
    [LoadColumn(3)]
    public float TripTime;
    [LoadColumn(4)]
    public float TripDistance;
    [LoadColumn(5)]
    public string PaymentType;
    [LoadColumn(6)]
    public float FareAmount;
}

public class ModelOutput
{
    [ColumnName("Score")]
    public float FareAmount;
}