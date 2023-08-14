{- Implementation of 2048 using the Store Comonad -}

class Functor w => Comonad w where
  extract :: w a -> a
  duplicate :: w a -> w (w a)
  duplicate = extend id
  extend :: (w a -> b) -> w a -> w b
  extend f = fmap f . duplicate

data Store s a = Store (s -> a) s

instance Show Grid where
  show = toString

instance Functor (Store s) where
  fmap f (Store g s) = Store (f . g) s

instance Comonad (Store s) where
  extract (Store f s) = f s
  duplicate (Store f s) = Store (Store f) s

data Direction = L | R | U | D deriving (Eq)
type Position = (Int, Int)
type Value = Int
type Grid = Store Position Value

maxSize :: Int
maxSize = 3

toDirectionalLists :: Direction -> Grid -> [[Value]]
toDirectionalLists dir (Store f s) =
  case dir of
    L -> [map (f . (, y)) [0..maxSize] | y <- [0..maxSize]]
    R -> [map (f . (, y)) [maxSize, maxSize - 1..0] | y <- [0..maxSize]]
    D -> [map (f . (x, )) [0..maxSize] | x <- [0..maxSize]]
    U -> [map (f . (x, )) [maxSize, maxSize - 1..0] | x <- [0..maxSize]]

toLists :: Grid -> [[Value]]
toLists (Store f s) = reverse [[f (x, y) | x <- [0..maxSize]] | y <- [0..maxSize]]
toString :: Grid -> String
toString = unlines . map (unwords . map show) . toLists

listGameRule :: [Value] -> [Value]
listGameRule x = let x' = listGameRule' (filter (/= 0) x) in x' ++ replicate (length x - length x') 0
  where listGameRule' [] = []
        listGameRule' [x] = [x]
        listGameRule' (x:y:xs)
          | x == y = x + y : listGameRule' xs
          | otherwise = x : listGameRule' (y:xs)

gameRule :: Direction -> Grid -> Value
gameRule dir (Store f s) = 
  case dir of
    L -> map listGameRule (toDirectionalLists dir (Store f s)) !! y !! x
    R -> reverse (map listGameRule (toDirectionalLists dir (Store f s)) !! y) !! x
    D -> map listGameRule (toDirectionalLists dir (Store f s)) !! x !! y
    U -> reverse (map listGameRule (toDirectionalLists dir (Store f s)) !! x) !! y
  where (x, y) = s

initalState :: Grid
initalState = Store initalHelper (0, 0)

initalHelper :: Position -> Value
initalHelper (0, 0) = 2
initalHelper (3, 3) = 4
initalHelper (x, y) = if x `elem` [0..maxSize] && y `elem` [0..maxSize] then 0 else -1

main :: IO ()
main = do


--TODO
-- 1. Add random tiles
-- 2. Add game over
-- 3. Add win
-- 4. Add score
-- 5. Add input