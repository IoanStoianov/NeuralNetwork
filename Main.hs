import           Streamly
import qualified Streamly.Prelude              as S
import           Text.Printf                    ( printf )
import           Control.DeepSeq                ( force )
import           Control.Monad.Trans.Maybe
import           Data.IDX

import           Data.Vector  hiding (force, map, zip)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed           as UV
import           Data.Matrix
import           Data.List.Split                ( chunksOf )

import NN

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
    -- v1 = A.fromVector' Par (Sz2 1 784) v0

toExpectedVal :: Int -> UV.Vector Double
toExpectedVal index = UV.unfoldr (\(n, i)-> if n == 0 then Nothing 
  else if n==i then Just (1,(n-1,i)) else Just(0,(n-1,i))) (10, index)

mnistStream :: Int -> FilePath -> FilePath -> IO (SerialT IO  ([UV.Vector Double], [UV.Vector Double]))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  -- Split data into batches
  let (vs, labs) = Prelude.unzip dta
      vs'   = chunksOf batchSize vs
      labs' = chunksOf batchSize labs
      dta'  = zip vs' labs'
  return $ S.fromList dta'




trainNet :: Monad m => [Layer Double] -> SerialT m (UV.Vector Double, UV.Vector Double) -> Double -> m ([Layer Double], Double)
trainNet net dataStream learnRate = do
      S.foldl' forwardAndBackward (net, learnRate) dataStream




-- forwardAndBackward :: ([Layer Double], Double) -> (Vector Double, Vector Double) -> ([Layer Double], Double)
forwardAndBackward :: ([Layer Double], Double) -> (UV.Vector Double, UV.Vector Double) -> ([Layer Double], Double)
forwardAndBackward (net, learnRate) (data1,data2)  = (Prelude.reverse newNet, learnRate)
  where (input, expected) = (unbox data1, unbox data2)
        (_, nextOutput) = Prelude.foldl forward (input,[]) net
        net' = Prelude.reverse net
        nextOutput' = Prelude.reverse nextOutput
        firstDelta = calculateError (Prelude.head nextOutput') expected
        (_, _, _, newNet) = Prelude.foldl backprop (nextOutput, firstDelta , learnRate, []) net

forward :: (Vector Double, [Vector Double]) -> Layer Double -> (Vector Double, [Vector Double])
forward (input, output) (Layer w b act) = (out, out : output)
  where activate = getActivation act
        matrix = elementwise (+) (multStd2 w (build1Dmatrix $V.toList input)) (build1Dmatrix $ V.toList b)
        out = activate (getCol 1 matrix)

backprop :: ([Vector Double], Vector Double, Double, [Layer Double]) -> Layer Double -> ([Vector Double], Vector Double, Double, [Layer Double])
backprop (h:out, delta, learnRate, newNet)  (Layer w b act)  = (out, nextDelta, learnRate, newLayer : newNet)
      where deltaVector = V.map (* learnRate)  $ multpVectors h delta
            newLayer = updateWeights (Layer w b act) deltaVector
            nextDelta = getInsideDelta newLayer deltaVector
            

-- errorChange output expected layer = 0
--   where error = calculateError output expected
--       change =


lastLayerDelata output expected = error
  where error = calculateError output expected


derivative :: Vector Double -> Vector Double
derivative = V.map (1 - ) 


calculateError :: Vector Double -> Vector Double -> Vector Double
calculateError = V.zipWith (\ a b -> 0.5* (a-b)*(a-b))


updateWeights :: Layer Double -> Vector Double -> Layer Double
updateWeights (Layer w b act) deltaVector =  (Layer newWeights bias act)
    where newWeights = elementwise (+) w ( buildTransponseMatrix deltaVector (nrows w))
          bias = V.zipWith (+) b deltaVector
    -- create matrix and transponse

getInsideDelta :: Layer Double -> Vector Double -> Vector Double
getInsideDelta (Layer w b act) del = V.fromList newDelta
  where newWeights = elementwise (*) w ( buildTransponseMatrix del (nrows w)) 
        (newDelta, _) = Prelude.foldl sumWeights ([],newWeights) [1..(ncols w)]

sumWeights :: Num a => ([a], Matrix a) -> Int -> ([a], Matrix a)
sumWeights (result,input) i = (result Prelude.++ [colCum],input)
  where colCum = V.sum (getCol i input)



-- main =  do
--   dataStream <- mnistStream 100 "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
--  --  S.head x

--   let [l1, l2, l3] = [784, 300, 10]
--   w1 <- genWeights (l1, l2)
--   b1 <- genBias l1
--   w2 <- genWeights (l2, l3)
--   b2 <- genBias l2
--   w3 <- genWeights (l3, l3)
--   b3 <- genBias l3



--  --  return 0
--   let net = [Layer w1 b1 Sigmoid, Layer w2 b2 Sigmoid, Layer w3 b3 Sigmoid]

--   trainNet net dataStream 0.1