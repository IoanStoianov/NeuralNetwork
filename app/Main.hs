module Main where

import Lib

import System.TimeIt

import           Prelude as P

import           Control.Monad
import           Streamly
import qualified Streamly.Prelude              as S
import           Text.Printf                    ( printf )
import           Control.DeepSeq                ( force )
import           Control.Monad.Trans.Maybe
import           Data.IDX

import           Data.Vector  hiding (force, map, zip)
import qualified Data.Vector                   as V
import qualified Data.Vector.Unboxed           as UV
import           Data.Matrix
import           Data.List.Split                ( chunksOf )


loadMNIST :: FilePath -> FilePath -> IO (Maybe [(UV.Vector Double, UV.Vector Double)])
loadMNIST fpI fpL = runMaybeT $ do
  i <- MaybeT $ decodeIDXFile fpI
  l <- MaybeT $ decodeIDXLabelsFile fpL
  d <- MaybeT . return $ force $ labeledIntData l i
  return $ map _conv d
 where
  _conv :: (Int, UV.Vector Int) -> (UV.Vector Double, UV.Vector Double)
  _conv (label, v) = (v0, toExpectedVal label)
   where
    v0 = UV.map ((`subtract` 0.5) . (/ 255) . fromIntegral) v


mnistStream :: Int -> FilePath -> FilePath -> IO (SerialT IO  ([UV.Vector Double], [UV.Vector Double]))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  -- Split data into batches
  let (vs, labs) = P.unzip dta
      vs'   = chunksOf batchSize vs
      labs' = chunksOf batchSize labs
      dta'  = zip vs' labs'
  return $ S.fromList dta'

trainNet :: [Layer Double] -> SerialT IO ([UV.Vector Double], [UV.Vector Double]) -> Double -> IO ([Layer Double], Double)
trainNet net dataStream learnRate = S.foldl' trainBatch (net, learnRate) dataStream

trainBatch :: ([Layer Double], Double) -> ([UV.Vector Double], [UV.Vector Double]) -> ([Layer Double], Double)
trainBatch (net, learnRate) (input,labels) = (newNet,learnRate)
  where pairs = zip input labels
        (newNet, _) = P.foldl forwardAndBackward (net, learnRate) pairs


testBatch :: [Layer Double] -> [(Vector Double, Vector Double)] -> IO ()
testBatch net batch = do
  let result = map (tras net) batch
  let x = P.foldl (\ sum (x, y) -> if x ==y then sum+1 else sum) 0 result
      len = P.length result
  putStr $ printf "Training accuracy %d/%d \n" (x::Int) len
  return()

tras :: [Layer Double] -> (Vector Double, Vector Double) -> (Int, Int)
tras net (input,expected) = (getProposition1 output, getProposition1 expected)
  where (output,_) = V.foldl' forward (input,[]) (V.fromList net)

repeatTrain :: Monad m => ([Layer Double], Double) -> ([UV.Vector Double], [UV.Vector Double]) -> m [Layer Double]
repeatTrain netData input = do
  let newInput = P.replicate 5000 input
  let (trainedNet,_) = V.foldl' trainBatch netData (V.fromList newInput)
  return trainedNet

  
main =  do
  dataStream <- mnistStream 10 "../data/train-images-idx3-ubyte" "../data/train-labels-idx1-ubyte"
  Just input <- S.head dataStream

  let [l1, l2, l3, o] = [784, 80, 30, 10]
  w1 <- genWeights (l1, l2)
  b1 <- genBias l2
  w2 <- genWeights (l2, l3)
  b2 <- genBias l3
  w3 <- genWeights (l3, o)
  b3 <- genBias o
  -- 
  let net = [Layer w1 b1 Sigmoid, Layer w2 b2 Sigmoid,  Layer w3 b3 Sigmoid]

  -- trainNet net dataStream 0.1
  trainedNet <- repeatTrain (net, 0.3) input
  -- let (trainedNet,_) = trainBatch (net, 0.1) input

  let (trainD, label) = input
  let merged = zip (map unbox trainD) (map unbox label)
  -- showOutput net (P.head merged)
  -- timeIt $ testBatch trainedNet merged
  timeIt $ testBatch trainedNet (P.take 10 merged)

  return trainedNet

