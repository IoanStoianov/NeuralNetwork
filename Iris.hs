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

loadMNIST :: FilePath -> FilePath -> IO (Maybe [(UV.Vector Float, UV.Vector Float)])
loadMNIST fpI fpL = runMaybeT $ do
  i <- MaybeT $ decodeIDXFile fpI
  l <- MaybeT $ decodeIDXLabelsFile fpL
  d <- MaybeT . return $ force $ labeledIntData l i
  return $ map _conv d
 where
  _conv :: (Int, UV.Vector Int) -> (UV.Vector Float, UV.Vector Float)
  _conv (label, v) = (v0, toExpectedVal label)
   where
    v0 = UV.map ((`subtract` 0.5) . (/ 255) . fromIntegral) v
    -- v1 = A.fromVector' Par (Sz2 1 784) v0

toExpectedVal :: Int -> UV.Vector Float
toExpectedVal index = UV.unfoldr (\(n, i)-> if n == 0 then Nothing 
  else if n==i then Just (1,(n-1,i)) else Just(0,(n-1,i))) (10, index)

mnistStream :: Int -> FilePath -> FilePath -> IO (SerialT IO  ([UV.Vector Float], [UV.Vector Float]))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  -- Split data into batches
  let (vs, labs) = Prelude.unzip dta
      vs'   = chunksOf batchSize vs
      labs' = chunksOf batchSize labs
      dta'  = zip vs' labs'
  return $ S.fromList dta'
    
func =  do
   x <- mnistStream 3 "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
   f1 x


f1 :: Monad m => SerialT m a -> m (Maybe a)
f1 a = do S.head a
-- -- f :: (Matrix Float, Matrix Float) 

f :: (Matrix Float, Matrix Float) -> p -> (Matrix Float, Matrix Float)
f a _ =  a


x=fromLists [ [1,2,3]  
          , [4,5,6]  
          , [7,8,9] ]

-- y :: UV.Vector Double

-- trainNet net dataStream = do

-- epochStep net = S.foldl' passForward net dataStream

forwardAndBackward net dataS = nextOutput
  where (input, expected) = dataS
        nextOutput = Prelude.foldl forward (input,[]) net
        -- backprop

forward :: (Vector Double, [Vector Double]) -> Layer Double -> (Vector Double, [Vector Double])
forward (input, output) (Layer w b act) = (out, out : output)
  where activate = getActivation act
        matrix = elementwise (+) (multStd2 w (build1Dmatrix $V.toList input)) (build1Dmatrix $ V.toList b)
        out = activate (getCol 1 matrix)


-- backprop output expected net =

calclulateError :: Vector Double -> Vector Double -> Vector Double
calclulateError = V.zipWith (\ a b -> 0.5* (a-b)*(a-b))