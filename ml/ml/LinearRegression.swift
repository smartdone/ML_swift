//
//  LinearRegression.swift
//  ml
//
//  Created by ssd on 2019/2/20.
//  Copyright © 2019 bangcle. All rights reserved.
//

import Foundation

// http://insideai.cn/start/lesson2.html
// 线性回归

func predict(x:Double, w:Double, b:Double) -> Double {
    let yhat:Double = w * x + b;
    return yhat
}

// 曼哈顿距离
func ManhattanDistance(num1:[Double], num2:[Double]) -> Double {
    if num1.count == num2.count{
        var loss:Double = 0
        for i in 0..<num1.count{
            loss += fabs(num1[i] - num2[i])
        }
        return loss
    }
    return 0
}

// 欧式距离 L = (yhat-y)2 一般用 L = 1/2*(yhat-y)2
func EuclideanDistance(num1:[Double], num2:[Double]) -> Double {
    if num1.count == num2.count{
        var loss:Double = 0
        for i in 0..<num1.count{
            loss += pow((num1[i] - num2[i]), 2) / 2
        }
        return loss
    }
    return 0
}


func testLossFunction(){
    let labels:[Double] = [10,10,10,10,10];
    let answers1:[Double] = [6, 9, 10, 11, 14];
    let answers2:[Double] = [6, 9, 9, 9, 7];
    let answers3:[Double] = [8, 8, 8, 8, 8];
    
    print("Manhattan Distance answers1: \(ManhattanDistance(num1: labels, num2: answers1))")
    print("Manhattan Distance answers2: \(ManhattanDistance(num1: labels, num2: answers2))")
    print("Manhattan Distance answers3: \(ManhattanDistance(num1: labels, num2: answers3))")
    
    print("Euclidean Distance answers1: \(EuclideanDistance(num1: labels, num2: answers1))")
    print("Euclidean Distance answers2: \(EuclideanDistance(num1: labels, num2: answers2))")
    print("Euclidean Distance answers3: \(EuclideanDistance(num1: labels, num2: answers3))")
}

func sampleLinearRegression() {
    let area = "area"
    let money = "money"
    var trainData = [[area:0.85,money:7.05],
                     [area:1.03,money:8.6],
                     [area:1.118,money:9.25],
                     [area:1.17,money:9.7],
                     [area:1.19,money:9.88],
                     [area:1.25,money:10.35],
                     [area:1.309,money:10.82]];
    
    var testData = [[area:0.92,money:7.65],
                    [area:1.08,money:8.95],
                    [area:1.15,money:9.55],
                    [area:1.2,money:9.95]];
    predict(x: trainData[0][area]!, w: 1, b: 0)
}
