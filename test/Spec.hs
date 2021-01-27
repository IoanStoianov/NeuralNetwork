import Test.HUnit

import Prelude as P

import Lib
import           Data.Vector as V
import qualified Data.Vector.Unboxed           as UV

import           Data.Matrix as M
 
buildTransponseMatrixTest :: Test
buildTransponseMatrixTest = TestCase $ do
    let vec = V.fromList [1,2,3,4,5]
        matr = buildTransponseMatrix vec 4
    assertEqual "BuildTransponseMatrix" matr (M.fromLists [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]])

toExpectedValTest :: Test
toExpectedValTest = TestCase $ do
    let expected = UV.fromList [0,0,0,0,0,0,1,0,0,0] :: UV.Vector Double
    assertEqual "Num to expected vector: 6" expected (toExpectedVal 6)

    let expected = UV.fromList [1,0,0,0,0,0,0,0,0,0] :: UV.Vector Double
    assertEqual "Num to expected vector: 0" expected (toExpectedVal 0)

    let expected = UV.fromList [0,0,0,0,0,0,0,0,0,1] :: UV.Vector Double
    assertEqual "Num to expected vector: 9" expected (toExpectedVal 9)

calculateErrorTest :: Test
calculateErrorTest = TestCase $ do
    let output = V.fromList [4, 9, 10] :: V.Vector Double
        real = V.fromList [2,6,5] :: V.Vector Double
        expected = V.fromList [2, 4.5, 12.5]:: V.Vector Double
    
    assertEqual "Calculate error" expected (calculateError output real)

sumColWeightsTest :: Test
sumColWeightsTest = TestCase $ do
    let matr = M.fromLists [[1,4,7], [2,5,8], [3,6,9]] :: M.Matrix Double
    let (result, _) = sumColWeights ([], matr) 1
    assertEqual "SumColWeights" 6 (P.head result)
    let (result, _) = sumColWeights ([], matr) 2
    assertEqual "SumColWeights" 15 (P.head result)
    let (result, _) = sumColWeights ([], matr) 3
    assertEqual "SumColWeights" 24 (P.head result)


updateWeightsTest :: Test
updateWeightsTest = TestCase $ do 
    let matr = M.fromLists [[4,4,4], [0,0,0], [0,0,0]] :: M.Matrix Double
        bias = V.fromList [1,1,1] :: V.Vector Double
        delta = V.fromList [1,2,3] :: V.Vector Double
        layer = Layer matr bias Id
    let expectedMatr = M.fromLists [[5,5,5], [2,2,2], [3,3,3]] :: M.Matrix Double
        expectedBias = V.fromList [2,3,4] :: V.Vector Double
    
    let (Layer newMatr newBias _) = updateWeights layer delta

    assertEqual "UpdateWeightsTest matrix" newMatr expectedMatr
    assertEqual "UpdateWeightsTest bias" expectedBias newBias

getInsideDeltaTest :: Test
getInsideDeltaTest = TestCase $ do
    let matr = M.fromLists [[1,1,1], [1,1,1], [1,1,1]] :: M.Matrix Double
        bias = V.fromList [1,1,1] :: V.Vector Double
        delta = V.fromList [1,2,3] :: V.Vector Double
        layer = Layer matr bias Id
    let result = getInsideDelta layer delta
        expected = V.fromList [6,6,6] :: V.Vector Double
    assertEqual "getInsideDeltaTest" expected result 

forwardTest :: Test
forwardTest = TestCase $ do
    let matr = M.fromLists [[1,2,3], [4,5,6], [7,8,9]] :: M.Matrix Double
        bias = V.fromList [1,1,1] :: V.Vector Double
        input = V.fromList [1,2,3] :: V.Vector Double
        layer = Layer matr bias Id

    let (result, _) = forward (input, []) layer
        expected = V.fromList [15,33,51] :: V.Vector Double

    assertEqual "ForwardTest" expected result

    let matr = M.fromLists [[1,2,3,1], [4,5,6,1], [7,8,9,1]] :: M.Matrix Double
        bias = V.fromList [1,1,1,1] :: V.Vector Double
        input = V.fromList [1,2,3,1] :: V.Vector Double
        layer = Layer matr bias Id

    let (result, _) = forward (input, []) layer
        expected = V.fromList [16,34,52] :: V.Vector Double

    assertEqual "ForwardTest" expected result

backpropTest :: Test
backpropTest = TestCase $ do
    let matr = M.fromLists [[1,2,3], [4,5,6], [7,8,9]] :: M.Matrix Double
        bias = V.fromList [1,1,1] :: V.Vector Double
        delta = V.fromList [1,2,3] :: V.Vector Double
        output = V.fromList [10,20,30] :: V.Vector Double -- 1,4,9
        layer = Layer matr bias Id
    let (_,newDelta,_,[Layer newW newB act]) = backprop ([output], delta, 0.1, []) layer
        expectedDelta  = V.fromList [178,192,206] :: V.Vector Double
        expectedWeight =  M.fromLists [[2,3,4], [8,9,10], [16,17,18]] :: M.Matrix Double
        expectedBias = V.fromList [2,5,10] :: V.Vector Double

    assertEqual "BackpropTest weights" expectedWeight newW
    assertEqual "BackpropTest bias" expectedBias newB
    assertEqual "BackpropTest delta" expectedDelta newDelta

tests :: Test
tests =
  TestList
    [ 
      buildTransponseMatrixTest,
      toExpectedValTest,
      calculateErrorTest,
      sumColWeightsTest,
      updateWeightsTest,
      getInsideDeltaTest,
      forwardTest,
      backpropTest
    ]

main :: IO Counts
main = runTestTT tests