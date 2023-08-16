{- Implementation of 2048 using the Store Comonad -}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
module Main where
import           Data.MemoCombinators as Memo

class Functor w => Comonad w where
  extract :: w a -> a
  duplicate :: w a -> w (w a)
  duplicate = extend id
  extend :: (w a -> b) -> w a -> w b
  extend f = fmap f . duplicate

data Store s a = Store (s -> a) s

instance (Show a, Show s) => Show (Store s a) where
  show (Store f s) = let g = Store f s in show s ++ " " ++ (show . extract) g

instance Functor (Store s) where
  fmap f (Store g s) = Store (f . g) s

instance Comonad (Store s) where
  extract (Store f s) = f s
  duplicate (Store f s) = Store (Store f) s

data Direction = L | R | U | D deriving (Eq)
type Position = (Int, Int)
type Value = Int
type Grid a = Store Position a

maxSize :: Int
maxSize = 2

tab :: Memo s -> Store s a -> Store s a
tab f (Store g s) = Store (f g) s

experiment :: Functor f => (s -> f s) -> Store s a -> f a
experiment f (Store g s) = fmap g (f s)

rowIndices :: Direction -> Position -> [Position]
rowIndices dir (x, y) =
  case dir of
    L -> [(x, y) | x <- [0..maxSize]]
    R -> [(x, y) | x <- [maxSize, maxSize - 1..0]]
    D -> [(x, y) | y <- [0..maxSize]]
    U -> [(x, y) | y <- [maxSize, maxSize - 1..0]]

toRow :: Direction -> Grid a -> [a]
toRow dir = experiment (rowIndices dir)

listGameRule :: [Value] -> [Value]
listGameRule x = let x' = listGameRule' (filter (/= 0) x) in x' ++ replicate (length x - length x') 0
  where listGameRule' [] = []
        listGameRule' [x] = [x]
        listGameRule' (x:y:xs)
          | x == y = x + y : listGameRule' xs
          | otherwise = x : listGameRule' (y:xs)

gameRule :: Direction -> Grid Value -> Value
gameRule dir (Store f s) =
  case dir of
    L -> listGameRule (toRow dir g) !! x
    R -> reverse (listGameRule (toRow dir g)) !! x
    D -> listGameRule (toRow dir g) !! y
    U -> reverse (listGameRule (toRow dir g)) !! y
  where (x, y) = s
        g = Store f s

initalState :: Grid Value
initalState = addNewTile $ addNewTile $ Store (const 0) (0, 0)

emptyTiles :: Grid Value -> [Position]
emptyTiles (Store f s) = filter (\x -> f x == 0) [(x, y) | x <- [0..maxSize], y <- [0..maxSize]]

addNewTile :: Grid Value -> Grid Value
addNewTile (Store f s) = Store (\x -> if x == (t !! (totalScore g `mod` length t) ) then 2 else f x) s
  where t = emptyTiles g
        g = Store f s

totalScore :: Grid Value -> Int
totalScore (Store f s) = sum [f (x, y) | x <- [0..maxSize], y <- [0..maxSize]]

isWon :: Grid Value -> Bool
isWon (Store f s) = 2048 `elem` ([f (x, y) | x <- [0..maxSize], y <- [0..maxSize]])

isLost :: Grid Value -> Bool
isLost (Store f s) = null (emptyTiles (Store f s))

endCondition :: Grid Value -> IO Bool
endCondition g 
  | isWon g = do
    putStrLn "You won!"
    return True
  | isLost g = do
    putStrLn "You lost!"
    return True
  | otherwise = return False

render :: Grid Value -> String
render g = unlines $ map (unwords . map show) $ toLists g
    where toLists (Store f s) = reverse [[f (x, y) | x <- [0..maxSize]] | y <- [0..maxSize]]

mainLoop :: Grid Value -> IO ()
mainLoop g = do
  print g
  putStrLn (render g)
  end <- endCondition g
  if end
    then return ()
  else do
    putStrLn "Enter a direction (l, r, u, d): "
    dir <- getLine
    case dir of
      "l" -> mainLoop (addNewTile $ extend (gameRule L . tab (Memo.pair Memo.integral Memo.integral)) g)
      "r" -> mainLoop (addNewTile $ extend (gameRule R . tab (Memo.pair Memo.integral Memo.integral)) g)
      "u" -> mainLoop (addNewTile $ extend (gameRule U . tab (Memo.pair Memo.integral Memo.integral)) g)
      "d" -> mainLoop (addNewTile $ extend (gameRule D . tab (Memo.pair Memo.integral Memo.integral)) g)
      _   -> mainLoop g

main :: IO ()
main = mainLoop initalState