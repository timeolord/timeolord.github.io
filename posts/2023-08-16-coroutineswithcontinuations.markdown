---
title: Coroutines in Haskell with Monadic Continuations
---

Coroutines are a powerful feature that can help you create responsive games. Coroutines are functions that can pause and resume their execution at any point, without blocking the main thread of the game. This allows you to perform tasks that require waiting, delays, or multiple steps, without affecting the responsiveness of your game. There are a perfect abstraction for modeling events over time, such as animations, physics, and AI.

For example, imagine you want to create a simple animation where a character moves from one point to another, while changing its color and size. You could write a coroutine that updates the character’s position, color, and size every frame, and pauses until the next frame using the yield statement, without worrying about an explicit time parameter. 

Coroutines are a common feature in many programming languages, such as Lua, Python, and C#. However, they are not a native feature of Haskell. In this article, we will explore how to implement coroutines in Haskell using continuations and monads. We will also see how to use coroutines to create a simple game.

## What is a continuation?

A continuation is a way of representing the state of a computation at any point in time. Like a snapshot of what the program is doing and what it needs to do next. A continuation can be used to resume the computation from where it left off, or to transfer the control to a different part of the program. A continuation can also be passed as an argument to another function, which can then decide how to continue the computation. This is called continuation-passing style, and it is a common technique in functional programming languages.

One way to understand continuations is to imagine that every function has an extra parameter that represents what the function should do after it finishes its work. This parameter is called the continuation, and it is usually a function that takes the result of the original function as its input. For example, suppose we have a function 
```haskell
id :: a -> a
id x = x
```
We can rewrite it in continuation-passing style as
```haskell
idCps :: a -> (a -> r) -> r
idCps x k = k x
```
where `k` is the continuation. The function `id_cps` calls `k` with `x`. Since continuations allow you to transfer the control to a different part of the program, we can use them to implement control flow constructs such as `if`. For example, we can implement `if` as
```haskell
ifCps :: Bool -> (a -> r) -> (a -> r) -> a -> r
ifCps True t f a = t a
ifCps False t f a = f a
```
where `t` is the continuation that will be called if the condition is true, and `f` is the continuation that will be called if the condition is false. We can use `ifCps` to implement a function that returns the absolute value of a number:
```haskell
absCps :: (Num r, Ord a, Num a) => a -> (a -> r) -> r
absCps x k = ifCps (x >= 0) k (\x -> k (-x)) x
```
which we can use as
```haskell
> absCps (-5) id
5
```

## Monadic continuations

Continuations are a powerful abstraction, but they are not very convenient to use. Continuations are functions that take a single argument, and return a value. This means that if we want to use continuations to implement a coroutine, we would have to pass the coroutine’s state as an argument to the continuation, and return the coroutine’s state as the result of the continuation. This is not very convenient, because it means that we would have to explicitly pass the coroutine’s state as an argument to every function that uses the coroutine.

Instead we can use the monad abstraction to simplify this process. 

## What is Monad?

A monad is a concept from category theory. A category is a collection of objects and arrows, called morphisms, between them, that satisfy some basic rules. For example, the category of sets has sets as objects and functions as arrows.

A functor is a way of mapping one category to another, preserving the structure of the objects and arrows. For example, the power set functor maps any set to its power set (the set of all subsets), and any function to the function that takes the image of each subset.

An endofunctor is a functor that maps a category to itself. For example, the power set functor is an endofunctor on the category of sets.

A monad is a special kind of endofunctor, that comes with two additional operations: unit and join. The unit operation takes any object in the category and returns an object in the image of the endofunctor. The join operation takes any object in the image of the endofunctor applied twice, and returns an object in the image of the endofunctor applied once. These operations have to satisfy some coherence conditions, similar to the ones for monoids (which are structures with a binary operation and an identity element).

One way to think about a monad is that it adds some extra layer of structure or context to the objects and arrows in the category. For example, the power set monad adds the structure of subsets and inclusion relations. The unit operation wraps an object into a singleton subset, and the join operation flattens a set of subsets into a single subset.

This abstraction is a perfect abstraction for our coroutines, since the continuation is the context that we must pass around to every function that uses the coroutine.

## Monadic continuations

We can use the monad abstraction to simplify the implementation of coroutines. We can define a monad that represents a continuation as
```haskell
newtype Cont r a = Cont { runCont :: (a -> r) -> r }

instance Functor (Cont r) where
    fmap f m = Cont $ \c -> runCont m (c . f)

instance Applicative (Cont r) where
    pure a = Cont ($ a)
    (<*>) = ap

instance Monad (Cont r) where
    m >>= k  = Cont $ \c -> runCont m $ \a -> runCont (k a) c
```
where `r` is the type of the result, and `a` is the type of the value. The `runCont` function takes a continuation as an argument, and returns the result of the continuation. We can use this monad to implement our previous functions `idCps`, `ifCps`, and `absCps` as
```haskell
idCps :: a -> Cont r a
idCps x = Cont $ \k -> k x

ifCps :: Bool -> Cont r a -> Cont r a -> Cont r a
ifCps True t f = t
ifCps False t f = f

absCps :: (Num r, Ord a, Num a) => a -> Cont r a
absCps x = ifCps (x >= 0) (idCps x) (idCps (-x))
```
which gives us the same results as before:
```haskell
> runCont (absCps (-5)) id
5
```
