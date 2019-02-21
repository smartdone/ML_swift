import Cocoa
import CreateML

let TYPE_INT = 0
let TYPE_STRING = 1
let missing = "missing" // 用来记录缺失的值

// 加载训练文件
let trainFile = Bundle.main.url(forResource: "train", withExtension: "csv")!
var trainData = try MLDataTable(contentsOf: trainFile)

func addItem(data :inout [String:Int], key: String){
    if data.keys.contains(key) {
        data.updateValue(data[key]! + 1, forKey: key)
    } else {
        data.updateValue(1, forKey: key)
    }
}

// 计算每个值出现的次数
func valueCounts(data: MLUntypedColumn, type: Int) -> [String: Int] {
    var vc = [String:Int]()
    for i in 0..<data.count {
        if data[i].isValid {
            if type == TYPE_INT {
                addItem(data: &vc, key: String(stringInterpolationSegment: data[i].intValue!))
            } else if type == TYPE_STRING {
                addItem(data: &vc, key: data[i].stringValue!)
            }
        } else {
            addItem(data: &vc, key: missing)
        }
    }
    return vc
}

func getMode(data: MLUntypedColumn, type: Int) -> String {
    let vc = valueCounts(data: data, type: type)
    return  getMode(data: vc)
}

func getMode(data: [String:Int]) -> String{
    var max:Int = 0
    var maxKey:String = ""
    var flag = false
    for (key, value) in data{
        if key == missing {
            continue
        } else {
            if !flag {
                max = value
                maxKey = key
                flag = true
            } else {
                if max < value {
                    max = value
                    maxKey = key
                }
            }
        }
    }
    return maxKey
}

var strColumns = [String]()
var intColumns = [String]()

for item in trainData.columnTypes.keys {
    if trainData.columnTypes[item]! == MLDataValue.ValueType.int {
        intColumns.append(item)
    }
    if trainData.columnTypes[item]! == MLDataValue.ValueType.string {
        strColumns.append(item)
    }
}

print("String columns: \(strColumns)")
print("int columns: \(intColumns)")

// 去除缺失70%以上的
var drop = [String]()


for key in strColumns {
    let vc = valueCounts(data: trainData[key], type: TYPE_STRING)
    if vc.keys.contains(missing) {
        if Double(vc[missing]!) / Double(trainData[key].count) > 0.7 {
            trainData.removeColumn(named: key)
            print("remove column \(key)")
            drop.append(key)
        }
    }
}

for key in intColumns {
    let vc = valueCounts(data: trainData[key], type: TYPE_INT)
    if vc.keys.contains(missing) {
        if Double(vc[missing]!) / Double(trainData[key].count) > 0.7 {
            trainData.removeColumn(named: key)
            print("remove column \(key)")
            drop.append(key)
        }
    }
}

print(trainData.columnNames.count)

// 填充数据
var strFill = [String: String]()
var intFill = [String: Int]()

for key in strColumns {
    if !drop.contains(key) {
        let mode = getMode(data: trainData[key], type: TYPE_STRING)
        let modeDatavalue = MLDataValue.string(mode)
        trainData = trainData.fillMissing(columnNamed: key, with: modeDatavalue)
        strFill.updateValue(mode, forKey: key)
    }
}

for key in intColumns {
    if !drop.contains(key) {
        let mean = trainData[key].ints?.mean()
        let meanDatavalue = MLDataValue.int(Int(mean!))
        trainData = trainData.fillMissing(columnNamed: key, with: meanDatavalue)
        intFill.updateValue(Int(mean!), forKey: key)
    }
}
trainData.removeColumn(named: "Id")

print(drop)
print(strFill)
print(intFill)

let (EvaluationTable, TrainingTable) = trainData.randomSplit(by: 0.20, seed: 5)
// trainingData: CreateML.MLDataTable, targetColumn: String, featureColumns: [String]? = default, parameters
let model = try MLLinearRegressor(trainingData: TrainingTable, targetColumn: "SalePrice")

let trainingError = model.trainingMetrics.maximumError
print("training Error: \(trainingError)")

let validationError = model.validationMetrics.rootMeanSquaredError
print("validation Error: \(validationError)")

let Evaluation = model.evaluation(on: EvaluationTable)
let evaluationError = Evaluation.rootMeanSquaredError
print("Evaluation Error: \(evaluationError)")

//let drop = ["Fence", "PoolQC", "Alley", "MiscFeature"]
//let strFill = ["LandSlope": "Gtl", "Functional": "Typ", "HeatingQC": "Ex", "GarageFinish": "Unf", "PavedDrive": "Y", "Exterior1st": "VinylSd", "LotShape": "Reg", "Neighborhood": "NAmes", "BldgType": "1Fam", "Exterior2nd": "VinylSd", "BsmtFinType2": "Unf", "Electrical": "SBrkr", "Heating": "GasA", "Condition2": "Norm", "Condition1": "Norm", "Street": "Pave", "HouseStyle": "1Story", "GarageCond": "TA", "FireplaceQu": "Gd", "SaleType": "WD", "GarageType": "Attchd", "KitchenQual": "TA", "MasVnrType": "None", "BsmtQual": "TA", "ExterQual": "TA", "LandContour": "Lvl", "Utilities": "AllPub", "RoofStyle": "Gable", "CentralAir": "Y", "SaleCondition": "Normal", "ExterCond": "TA", "GarageQual": "TA", "BsmtCond": "TA", "RoofMatl": "CompShg", "BsmtFinType1": "Unf", "BsmtExposure": "No", "LotConfig": "Inside", "Foundation": "PConc", "MSZoning": "RL"]
//let intFill = ["1stFlrSF": 1162, "Id": 730, "ScreenPorch": 15, "GarageCars": 1, "WoodDeckSF": 94, "MoSold": 6, "TotalBsmtSF": 1057, "BsmtFinSF1": 443, "LotArea": 10516, "BsmtHalfBath": 0, "YrSold": 2007, "FullBath": 1, "MasVnrArea": 103, "BsmtFullBath": 0, "YearRemodAdd": 1984, "LowQualFinSF": 5, "BedroomAbvGr": 2, "LotFrontage": 70, "PoolArea": 2, "GarageYrBlt": 1978, "HalfBath": 0, "GarageArea": 472, "MiscVal": 43, "OverallQual": 6, "EnclosedPorch": 21, "YearBuilt": 1971, "SalePrice": 180921, "TotRmsAbvGrd": 6, "Fireplaces": 0, "3SsnPorch": 3, "2ndFlrSF": 346, "MSSubClass": 56, "OpenPorchSF": 46, "BsmtUnfSF": 567, "KitchenAbvGr": 1, "OverallCond": 5, "GrLivArea": 1515, "BsmtFinSF2": 46]
//let strColumns = ["FireplaceQu", "Utilities", "GarageType", "MiscFeature", "LotConfig", "LandSlope", "Condition1", "SaleCondition", "LandContour", "LotShape", "Exterior2nd", "BsmtExposure", "MSZoning", "Functional", "GarageFinish", "MasVnrType", "SaleType", "BsmtCond", "Exterior1st", "Heating", "RoofStyle", "HouseStyle", "Condition2", "GarageQual", "ExterQual", "Foundation", "BsmtFinType2", "ExterCond", "GarageCond", "Alley", "CentralAir", "Fence", "HeatingQC", "RoofMatl", "PavedDrive", "Electrical", "BldgType", "Neighborhood", "BsmtQual", "BsmtFinType1", "Street", "KitchenQual", "PoolQC"]
//let intColumns = ["MoSold", "BsmtFinSF2", "YearRemodAdd", "BsmtFullBath", "MSSubClass", "GrLivArea", "HalfBath", "BedroomAbvGr", "SalePrice", "1stFlrSF", "TotRmsAbvGrd", "EnclosedPorch", "LotArea", "2ndFlrSF", "LotFrontage", "WoodDeckSF", "ScreenPorch", "OpenPorchSF", "FullBath", "LowQualFinSF", "MiscVal", "GarageYrBlt", "PoolArea", "KitchenAbvGr", "GarageCars", "BsmtUnfSF", "TotalBsmtSF", "YrSold", "BsmtFinSF1", "Fireplaces", "Id", "3SsnPorch", "OverallQual", "YearBuilt", "BsmtHalfBath", "MasVnrArea", "OverallCond", "GarageArea"]

let testFile = Bundle.main.url(forResource: "test", withExtension: "csv")!
var testData = try MLDataTable(contentsOf: testFile)

print(testData)
for key in drop{
    testData.removeColumn(named: key)
}

for key in strColumns {
    if key == "SalePrice"{
        continue
    }
    if !drop.contains(key) {
        let mode = strFill[key]
        let modeDatavalue = MLDataValue.string(mode!)
        testData = testData.fillMissing(columnNamed: key, with: modeDatavalue)
    }
}
for key in intColumns {
    if key == "SalePrice"{
        continue
    }
    if !drop.contains(key) {
        let mean = intFill[key]
        let meanDatavalue = MLDataValue.int(Int(mean!))
        testData = testData.fillMissing(columnNamed: key, with: meanDatavalue)
    }
}

testData.removeColumn(named: "Id")
//print(testData)
var predict = try model.predictions(from: testData)
let sampleFile = Bundle.main.url(forResource: "sample_submission", withExtension: "csv")!
let sampleData = try MLDataTable(contentsOf: sampleFile)
print(sampleData)
let fileManager = FileManager.default
let fileName = "/result.csv"
let file = NSSearchPathForDirectoriesInDomains(FileManager.SearchPathDirectory.documentDirectory, FileManager.SearchPathDomainMask.userDomainMask, true).first
let path = file! + fileName

fileManager.createFile(atPath: path, contents:nil, attributes:nil)

let handle = FileHandle(forWritingAtPath:path)
var message = "Id,SalePrice\n"

for i in 0..<sampleData["Id"].count{
    let line = String(stringInterpolationSegment: sampleData["Id"][i].intValue!)
    let pre = String(stringInterpolationSegment: predict[i].doubleValue!)
    message += line + "," + pre + "\n"
}

handle?.write(message.data(using: String.Encoding.utf8)!)
print(path)

