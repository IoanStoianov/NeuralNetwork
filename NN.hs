module NN where

import           Control.Monad                  ( replicateM, foldM)
import           Control.Applicative            ( liftA2 )
-- import qualified System.Random                 as R
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

build1Dmatrix ::  [Double] -> Matrix Double
build1Dmatrix input = fromLists [input]
