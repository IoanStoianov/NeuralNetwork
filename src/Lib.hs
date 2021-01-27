module Lib where

import           Prelude as P

import Numeric

import           Control.Monad                  
import           Control.Applicative            ( liftA2 )
import qualified System.Random                 as R

import           Data.List                      ( maximumBy )
import           Data.Ord
import           Streamly
import qualified Streamly.Prelude              as S

import           Data.Maybe                     ( fromMaybe )


import           Data.IDX
import           Data.Vector                   as V
import qualified Data.Vector.Unboxed           as UV 
import           Data.Matrix   


data Layer a = 
    -- weight and bias
        Layer (Matrix a) (Vector a) Activation
            -- | Batchnorm (UV.Vector a) (UV.Vector a) (UV.Vector a) (UV.Vector a) Activation

data Activation = Relu | Sigmoid | Id

type NeuralNetwork a = [Layer a]

getActivation :: Activation -> (Vector Double -> Vector Double)
getActivation Id = id
getActivation Sigmoid = V.map sigmoid
getActivation Relu = V.map relu

-- multWeights input weight = 

-- buildInputMatrix:: UV.Vector Double -> Matrix Double
-- buildInputMatrix input = fromLists list
--     where len = UV.length input
--           list = P.replicate len (UV.toList input)

sigmoid :: Double -> Double
sigmoid x = recip $ 1.0 + exp (-x)

relu :: Double -> Double
relu x = if x < 0 then 0 else x


buildTransponseMatrix:: Vector Double -> Int -> Matrix Double
buildTransponseMatrix input len = transpose(fromLists list)
    where  list = P.replicate len (V.toList input)

build1Dmatrix ::  Vector Double -> Matrix Double
build1Dmatrix = colVector

multpVectors :: Vector Double -> Vector Double -> Vector Double
multpVectors vec1 vec2 = if V.null vec2 then vec1 else V.zipWith (*) vec1 vec2

genWeights :: (Int, Int) -> IO (Matrix Double)
genWeights (x, y) = fmap fromLists rand
    where
    rand = Control.Monad.replicateM y (getRandomList x)
    
genBias len = fmap V.fromList (getRandomList len)

getRandomList :: Int -> IO [Double]
getRandomList len = Control.Monad.replicateM len getRandomWeight

getRandomWeight :: IO Double
getRandomWeight =   (/100) <$> R.getStdRandom (R.randomR (-5,5))

unbox:: UV.Vector Double -> Vector Double
unbox vec = V.fromList (UV.toList vec)


toExpectedVal :: Int -> UV.Vector Double
toExpectedVal index = UV.unfoldr (\(n, i)-> if n == 10 then Nothing 
  else if n==i then Just (1,(n+1,i)) else Just(0,(n+1,i))) (0, index)


derivative :: Vector Double -> Vector Double
derivative = V.map (1 - ) 

calculateError :: Vector Double -> Vector Double -> Vector Double
calculateError = V.zipWith (\ a b -> b-a)

updateWeights :: Layer Double -> Vector Double -> Layer Double
updateWeights (Layer w b act) deltaVector =  Layer newWeights bias act
  where newWeights = elementwise (+) w ( buildTransponseMatrix deltaVector (ncols w))
        bias = V.zipWith (+) b deltaVector

getInsideDelta :: Layer Double -> Vector Double -> Vector Double
getInsideDelta (Layer w _ act) delta = V.fromList newDelta
  where newWeights = elementwise (*) w ( buildTransponseMatrix delta (ncols w)) 
        (newDelta, _) = V.foldl' sumColWeights ([],newWeights) (V.fromList [1..(ncols w)])


sumColWeights :: ([Double], Matrix Double) -> Int -> ([Double], Matrix Double)
sumColWeights (result,input) i = (result P.++ [colSum],input)
  where colSum = V.sum (getCol i input)

forward :: (Vector Double, [Vector Double]) -> Layer Double -> (Vector Double, [Vector Double])
forward (input, output) (Layer w b act) = (out, out : output)
  where activate = getActivation act
        matrix = elementwise (+) (multStd2 w (build1Dmatrix input)) (build1Dmatrix b)
        out = activate (getCol 1 matrix)

backprop :: ([Vector Double], Vector Double, Double, [Layer Double]) -> Layer Double -> ([Vector Double], Vector Double, Double, [Layer Double])
backprop (h:output, error, learnRate, newNet)  (Layer w b act)  = (output, nextDelta, learnRate, newLayer : newNet)
  where deltaVec = V.map (* learnRate)  $ multpVectors h error
        deltaVector = multpVectors deltaVec (derivative h)
        deltaVector2 = multpVectors deltaVector (P.head output)
        newLayer = updateWeights (Layer w b act) deltaVector2
        nextDelta = getInsideDelta newLayer deltaVector2

forwardAndBackward :: ([Layer Double], Double) -> (UV.Vector Double, UV.Vector Double) -> ([Layer Double], Double)
forwardAndBackward (net, learnRate) (trainData,labels)  = (newNet, learnRate)
  where (input, expected) = (unbox trainData, unbox labels)
        (_, nextOutput) = V.foldl' forward (input,[V.empty]) (V.fromList net)
        net' = P.reverse net
        firstDelta = calculateError (P.head nextOutput) expected
        (_, _, _, newNet) = V.foldl' backprop (nextOutput, firstDelta , learnRate, []) (V.fromList net')


showOutput :: [Layer Double] -> Vector Double -> Vector Double -> IO ()
showOutput net input expected = do
  let (nextOutput, _) = V.foldl' forward (input,[]) (V.fromList net)
  print (V.map showFullPrecision nextOutput)
  print (V.map showFullPrecision expected)


round2 :: Double -> Double
round2 x = fromIntegral (round $ x * 1e2) / 1e2

showFullPrecision :: Double -> String
showFullPrecision x = showFFloat Nothing (round2 x) ""