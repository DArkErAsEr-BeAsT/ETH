# FMFP Things to Know



### Typing Proof in Mini-haskell:

-  when the first rule application is the two-term side by side rule 
($(…)::t_0$) then have to go up until the next rule application is *Abs*

### Defining the Fold function on Trees:

- type : (a->b) -> (a ((only if Node takes an argument a else no)) -> n times b after that depending on the number of Trees in the definition -> then once Tree a again -> b

- function definition : 

  ```haskell
  foldTree l n = go
      where go
          go (Leaf a) = l a
          go (Node a t1 t2) = n a (go t1) (go t2)
  ```

- mapTree definition:

  ```haskell
  mapTree f tree = foldTree leaf node tree where
    leaf x = Leaf (f x)
    node x t1 t2 = Node (f x) t1 t2
  ```

### Induction Proof

- If the lemma has a number in it (i.e. 0) then first prove the general lemma by replacing the number by *n*, then you can use the general lemma to prove the particular case.
- 