module NN where

import           Control.Monad                  ( replicateM, foldM)
import           Control.Applicative            ( liftA2 )
import qualified System.Random                 as R
-- import           System.Random.MWC              ( createSystemRandom )
-- import           System.Random.MWC.Distributions
--     ( standard )
import           Data.List                      ( maximumBy )
import           Data.Ord
import           Streamly
import qualified Streamly.Prelude              as S

import           Data.Maybe                     ( fromMaybe )


import           Data.IDX
import           Data.Vector
import qualified Data.Vector.Unboxed           as V 
import           Data.Matrix   

import    qualified       Numeric.LinearAlgebra as LA 

data Layer a = 
    -- weight and bias
        Layer (Matrix a) (Vector a) Activation
            -- | Batchnorm (V.Vector a) (V.Vector a) (V.Vector a) (V.Vector a) Activation

data Activation = Relu | Sigmoid | Id

type NeuralNetwork a = [Layer a]

getActivation :: Activation -> (Vector Double -> Vector Double)
getActivation Id = id
-- getActivation Sigmoid = sigmoid
-- getActivation Relu = relu

-- multWeights input weight = 

buildInputMatrix:: V.Vector Double -> Matrix Double
buildInputMatrix input = fromLists list
    where len = V.length input
          list = Prelude.replicate len (V.toList input)


buildTransponseMatrix:: Vector Double -> Int -> Matrix Double
buildTransponseMatrix input len = transpose(fromLists list)
    where  list = Prelude.replicate len (Data.Vector.toList input)

build1Dmatrix ::  [Double] -> Matrix Double
build1Dmatrix input = fromLists [input]

multpVectors :: Vector Double -> Vector Double -> Vector Double
multpVectors = Data.Vector.zipWith (*)



-- genWeights :: (Int, Int) -> IO (Matrix Double)
genWeights :: (a, Int) -> IO (Matrix Double)
genWeights (x, y) = fmap fromLists rand
    where
    rand = Prelude.mapM  (\x -> x) . Prelude.take y $ repeat (getRandomList y)
    
genBias len = fmap Data.Vector.fromList (getRandomList len)

getRandomList :: Int -> IO [Double]
getRandomList len = Prelude.mapM  (\x -> x) (Prelude.take len $ repeat getRandomWeight)

getRandomWeight :: IO Double
getRandomWeight =   (/10) <$> R.getStdRandom (R.randomR (-5,5))

unbox:: V.Vector Double -> Vector Double
unbox vec = Data.Vector.fromList (V.toList vec)

