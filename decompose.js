// Heuristic non-negative tensor decomposition using the symmetric higher order power method (S-HOPM).
// This works best if the tensor contains no negative components whatsoever. If it does contain them,
// it may find a suboptimal solution. In practice, as long as the negative components are small, this
// should not be a problem.
// TODO: Generalize to higher dimensions (much of it is already general, but specifically the frame building part is very 2D specific).
//       Two things depend on having a frame: ensuring convergence of the power method (although this uses a heuristic, so it might not even work, or be too extreme), and finding an initialization for the power method.
//       Another thing that might have to be changed when going to higher dimensions is the way the identity tensor is used.

var assert = require("assert")

module.exports = decompose
module.exports.makeTensorDot = makeTensorDot
module.exports.makeTensorFromVector = makeTensorFromVector
module.exports.makePartialInnerProduct = makePartialInnerProduct
module.exports.constructIdentityTensor = constructIdentityTensor

function decompose(A, eps) {
  if (eps === undefined) eps = 1e-6
  var epsSqr = eps*eps
  var epsVSqr = Math.pow(1e-8, 2)
  var epsLambda = eps
  var epsLambdaSqr = epsLambda*epsLambda

  var n = A.length-1
  var d = 2
  var tensorDot = makeTensorDot(n, d)
  var tensorFromVector = makeTensorFromVector(n, d)
  var tensorFromVectorNminOne = makeTensorFromVector(n-1, d)
  var partialInnerProduct = makePartialInnerProduct(n, n-1, d)
  var F = constructFrame(53, tensorFromVector)
  var Fv = constructVecs(53, tensorFromVector)
  var Id = constructIdentityTensor(n, d)

  // Find initial decomposition
  var Ar, vobj, v, grade, v_t, lambda, vList = [], gradeList = [], v_tList = [], lambdaList = []
  var i, oldv, ind, Apr, iter, numconverged
  var VA, Vl, VVl//, lambdaDiffSqr
  Ar = A.slice()
  while(tensorDot(Ar,Ar) > epsSqr) {
    // Find best rank one approximation using S-HOPM
    vobj = bestApprox(Ar, undefined, epsVSqr, F, Fv, Id, tensorFromVector, tensorDot, tensorFromVectorNminOne, partialInnerProduct)
    lambda = vobj.lambda
    if (lambda < epsLambda) break // As soon as we get a value that is too small or NEGATIVE, just abort
    vList.push(vobj.v)
    gradeList.push(vobj.grade)
    v_tList.push(vobj.v_t)
    lambdaList.push(lambda)

    // Now refine the decomposition using a "leave one out" scheme
    if (vList.length>1) {
      ind = 0
      for(iter=0; iter<10000; iter++) {
        if (ind === 0) numconverged = 0
        // Find best replacement for vList[ind]
        Ar = newZeroArray(A.length)
        for(i=0; i<v_tList.length; i++) {
          if (i===ind) continue
          vec_muladdeq(Ar, lambdaList[i], v_tList[i])
        }
        Ar = vec_sub(A,Ar) // Residue after subtracting everything except the current vector
        oldv = vList[ind]
        vobj = bestApprox(Ar, gradeList[ind], epsVSqr, F, Fv, Id, tensorFromVector, tensorDot, tensorFromVectorNminOne, partialInnerProduct) // Keep grade
        v = vobj.v
        if (v === undefined) console.log([vobj, oldv, gradeList[ind]])
        lambda = Math.max(0, vobj.lambda)
        if (vec_diffo_sqr(v, oldv) < epsVSqr) { // Also checks for v = -oldv
          numconverged++
          if (numconverged === vList.length) {
            //console.log("Breaking refinement after " + (iter+1) + " iterations.")
            break
          }
        }
        vList[ind] = v
        v_tList[ind] = vobj.v_t
        lambdaList[ind] = lambda
        ind = (ind+1) % vList.length // cycle through indices
      }
      if (numconverged < vList.length) {
        console.warn("Refinement failed.")
      }
      // Minimize residual using non-negative least squares based on multiplicative updates
      // We minimize \|A - V\lambda\|^2, using the update rule \lambda'_i = \lambda_i (V^T A)_i / (V^TV\lambda)_i
      // Note that this relies on the (seemingly reasonable) idea that V^T A should not contain negative values, given that A is assumed to be (almost) positive semi-definite.
      // V^TV must have only non-negative entries, since we only use positive semi-definite tensors for building V (at worst we could have round-off errors causing some very small negative values, but these are unlikely to cause problems, as VVl always also contains a component corresponding to the product of the current value of lambda and the norm of the tensor in question.
      VA = new Array(v_tList.length)
      for(i=0; i<v_tList.length; i++) {
        VA[i] = tensorDot(A, v_tList[i])
        if (VA[i] < 0) console.warn("VA[i] < 0")
        VA[i] = Math.max(0, VA[i]) // This is not really supposed to happen, but in case it does, let's try to prevent oscillations (instead, the corresponding lambda will probably be set to zero)
      }
      VVl = new Array(v_tList.length)
      for(nnlsiter=0; nnlsiter<1000; nnlsiter++) {
        Vl = newZeroArray(A.length)
        for(i=0; i<v_tList.length; i++) {
          vec_muladdeq(Vl, lambdaList[i], v_tList[i])
        }
        //lambdaDiffSqr = 0
        for(i=0; i<v_tList.length; i++) {
          VVl[i] = tensorDot(Vl, v_tList[i])
          lambda = lambdaList[i] * (VA[i] + epsLambda*1e-6) / (VVl[i] + epsLambda*1e-6)
          //lambdaDiffSqr += Math.pow(lambda - lambdaList[i], 2)
          lambdaList[i] = lambda
          if (VVl[i] < 0) console.warn("VVl[i] < 0")
        }
        //if (lambdaDiffSqr < epsLambdaSqr) break
      }
      // TODO: In theory it should be possible to remove vectors whose lambda has decreased to below some threshold, but it's a little tricky to then guarantee that we do not end up in an infinite loop.
    }

    // Determine residual for next iteration
    Ar = newZeroArray(A.length)
    for(i=0; i<v_tList.length; i++) {
      vec_muladdeq(Ar, lambdaList[i], v_tList[i])
    }
    Ar = vec_sub(A,Ar) // Residue after subtracting everything except the current vector
  }

  // Check error
  if (tensorDot(Ar,Ar) > epsSqr) {
    console.warn("Did not reach the requested error bound. I got: " + Math.sqrt(tensorDot(Ar,Ar)))
  }

  // Return coefficients and vectors
  return {lambda: lambdaList, v: vList, grade: gradeList}
}

function bestApprox(A, grade, epsSqr, F, Fv, Id, tensorFromVector, tensorDot, tFVNMO, pip) {
  var v, v_t, lambda, v2, v2_t, lambda2, bestErr, err, Ar, newgrade = grade
  if (grade === 1 || grade === undefined) {
    v = bestRankOne(A, epsSqr, F, Fv, Id, tensorDot, tFVNMO, pip)
    v_t = tensorFromVector(v)
    lambda = tensorDot(A, v_t)
    if (grade === undefined) {
      Ar = vec_muladdeq(A.slice(), -lambda, v_t)
      bestErr = tensorDot(Ar, Ar)
    }
    newgrade = 1
  }
  if (grade === 2 || grade === undefined) {
    v2 = []
    v2_t = Id
    lambda2 = tensorDot(A, v2_t)/tensorDot(v2_t, v2_t)
    if (grade === undefined) {
      Ar = vec_muladdeq(A.slice(), -lambda2, v2_t)
      err = tensorDot(Ar, Ar)
    }
    if (grade === 2 || err < bestErr) {
      v = v2
      v_t = v2_t
      lambda = lambda2
      newgrade = 2
    }
  }
  return {v: v, v_t: v_t, lambda: lambda, grade: newgrade}
}

function bestRankOne(A, epsSqr, F, Fv, Id, tensorDot, tFVNMO, pip) { // Symmetric higher order power method
  var v3, vnorm, oldv
  var pdres = makePositiveDefinite(A, F, Fv, Id, tensorDot) // should ensure convergence
  A = pdres.A
  var v = pdres.v // TODO: Allow different initializations/suggestions?
  for(var i=0; i<100; i++) { // TODO: Accelerate convergence
    v3 = tFVNMO(v)
    oldv = v
    v = pip(A,v3) // "Partial inner product" between a degree-n tensor and a degree-(n-1) tensor
    vnorm = Math.sqrt(v[0]*v[0] + v[1]*v[1])
    if (vnorm<1e-20) { // Looks like we've chosen a direction that is almost in the "null-space" of A, so let's try another
      do {
        do { // rejection sampling to get a new vector
          v = [Math.random()*2 - 1, Math.random()*2 - 1]
          vnorm = Math.sqrt(v[0]*v[0] + v[1]*v[1])
        } while(vnorm<1e-3 || vnorm>1)
        v[0] /= vnorm
        v[1] /= vnorm
      } while(Math.abs(v[0]*oldv[0] + v[1]*oldv[1]) > 0.5) // Find a vector that has an angle with oldv of at least 60 degrees
      continue
    }
    v[0] /= vnorm
    v[1] /= vnorm
    if (Math.pow(v[0]-oldv[0],2)+Math.pow(v[1]-oldv[1],2) < epsSqr || Math.pow(v[0]+oldv[0],2)+Math.pow(v[1]+oldv[1],2) < epsSqr) { // Also check for "oscillating" convergence (in principle we can end up on the opposite vector)
      //console.log("Breaking power method after " + (i+1) + " iterations.")
      return v
    }
  }
  var err = Math.min(Math.pow(v[0]-oldv[0],2)+Math.pow(v[1]-oldv[1],2), Math.pow(v[0]+oldv[0],2)+Math.pow(v[1]+oldv[1],2))
  console.warn("Symmetric higher order power method failed! Remaining error^2: " + err + " > " + epsSqr)
  return v
}

function makePositiveDefinite(A, F, Fv, Id, tensorDot) { // Adds the identity tensor so that the square matrix unfolding of A becomes (approximately) positive definite
  var i, dot, mindot = Infinity, maxdot = -Infinity, maxdir
  for(i=0; i<F.length; i++) {
    dot = tensorDot(F[i], A)
    mindot = Math.min(mindot, dot)
    if (dot>maxdot) {
      maxdot = dot
      maxdir = Fv[i]
    }
  }
  mindot -= maxdot - mindot // HEURISTIC (just having A PD is not enough to ensure that the square matrix unfolding is also PD, so far, this does appear to work...)
  mindot = Math.min(mindot, 0) // never remove something
  mindot = -mindot // Invert so we can muladd
  return {A: vec_muladd(A, mindot, Id), v: maxdir}
}

function constructIdentityTensor(n, d) {
  // Generate an identity tensor.
  // The degree-2 identity tensors are easy, they correspond to diagonal matrices (so indices of the form [1,1], [2,2], etc.).
  // Higher degree identity tensors are a bit more difficult, they are (can be) formed by symmetrized tensor powers of the degree-2 identity tensor.
  // For the 2D situation this looks like the following:
  //   I4: sym(1111 + 1122 + 2211 + 2222) = 1111 + 1122*2/6 + 2222
  //   I6: sym(111111 + 111122 + 112211 + 112222 + 221111 + 221122 + 222211 + 222222) = 111111 + 111122*3/15 + 112222*3/15 + 222222
  // The coefficients are equal to the number of (distinct) permutations possible for the /adjacent pairs/ of indices, divided by the number of (distinct) permutations possible for the indices.
  // The first is (for 2D) \binom{n/2}{m} (with m the number of pairs of two's), the second \binom{n}{2m}.
  // Using multinomials we get something like Multinomial[m_1, m_2, m_3]/Multinomial[2 m_1, 2 m_2, 2 m_3].
  // Suppose the current numbers of pairs are m_1, m_2, etc. and we increase m_1 by one, while decreasing m_2 by one.
  // We then have: (Multinomial[m_1+1, m_2-1, m_3]/Multinomial[2 m_1 + 2, 2 m_2 - 2, 2 m_3])/(Multinomial[m_1, m_2, m_3]/Multinomial[2 m_1, 2 m_2, 2 m_3]) = (2 m_1 + 1)/(2 m_2 - 1)
  assert((typeof n) === "number" || n instanceof Number, "n must be a number")
  assert((typeof d) === "number" || d instanceof Number, "n must be a number")
  assert((n%2)===0, "Identity tensors can only have even degrees!")
  assert((d%1)===0, "The dimension must be an integer!")
  assert(n>0 && d>0, "n and d must be positive")
  
  function coefUpdate(i1, i2) { // Update coef and indexpairHist
    var m1 = indexpairhist[i1]
    var m2 = indexpairhist[i2]
    num = num*m2/(m1+1)
    denom = denom*m2*(2*m2-1)/((2*m1+1)*(m1+1))
    indexpairhist[i1] += 1
    indexpairhist[i2] -= 1
  }
  
  // Gather coefficients for all (non-decreasing) sequences of /pairs/ of indices in lexicographic order
  var nh = n>>1
  var indices = new Int32Array(n/2), indexpairhist = new Int32Array(d), num = 1, nums = [num], denom = 1, denoms = [denom]
  indexpairhist[0] = nh // Initially we have n/2 pairs of index zero
  while(Math.min.apply(null, indices)<d-1) { // Loop while something can be increased
    for(var i=nh; i-->0;) { // Find right-most index that can be increased
      if (indices[i]<d-1) break
    }
    coefUpdate(indices[i]+1, indices[i])
    indices[i]++ // This is guaranteed to work, because the while entry condition guarantees the for loop will find some index to increase
    for(var j=i+1; j<nh; j++) { // Make sure rest of the indices are equal to indices[i] (lowest non-decreasing sequence with this prefix)
      if (indices[j] !== indices[i]) coefUpdate(indices[i], indices[j])
      indices[j] = indices[i]
    }
    nums.push(num) // Push onto list with numerators
    denoms.push(denom) // Denominator
  }
  
  // Gather coefficients for all (non-decreasing) sequences of indices in lexicographic order
  var indices = new Int32Array(n), coefficients = [nums.shift()/denoms.shift()], isListOfPairs
  indexpairhist[0] = n>>1 // Initially we have n/2 pairs of index zero
  while(Math.min.apply(null, indices)<d-1) { // Loop while something can be increased
    for(var i=n; i-->0;) { // Find right-most index that can be increased
      if (indices[i]<d-1) break
    }
    indices[i]++ // This is guaranteed to work, because the while entry condition guarantees the for loop will find some index to increase
    for(var j=i+1; j<n; j++) { // Make sure rest of the indices are equal to indices[i] (lowest non-decreasing sequence with this prefix)
      indices[j] = indices[i]
    }
    isListOfPairs = true // Go and figure out whether we have a list of pairs (can probably be done more efficiently using increased bookkeeping)
    for(var i=0; i<n; i+=2) {
      isListOfPairs = isListOfPairs && indices[i] === indices[i+1]
    }
    if (isListOfPairs) {
      coefficients.push(nums.shift()/denoms.shift())
    } else {
      coefficients.push(0)
    }
  }
  
  return coefficients
}

function makeTensorDot(n, d) {
  var indcnt = getIndicesAndCounts(n, d)
  var counts = indcnt.counts
  var terms = []
  for(var i=0; i<counts.length; i++) {
    terms.push(counts[i] === 1 ? ("A[" + i + "]*B[" + i + "]") : (counts[i] + "*A[" + i + "]*B[" + i + "]"))
  }
  return new Function("A", "B", "return " + terms.join("+"))
}

function makeTensorFromVector(n, d) {
  // Gives non-redundant components of symmetric tensor resulting from raising vec to fourth order, in lexicographic order, using the square roots of binomial coefficients.
  // This weighting preserves dot products when using the trivial unweighted dot product on the components.
  // This is accomplished by lexicographically enumerating all non-decreasing index sequences.

  var allIndices = getIndicesAndCounts(n, d).indices
    
  // Construct function
  var entries = [], entryFactors
  for(var ind=0; ind<allIndices.length; ind++) {
    entryFactors = []
    for(var i=0; i<n; i++) {
      entryFactors.push("vec[" + allIndices[ind][i] + "]")
    }
    entries.push(entryFactors.join("*"))
  }
  return Function("vec", "return [" + entries.join(",") + "]")
}

function makePartialInnerProduct(n1, n2, d) {
  // Compute the symmetrized partial inner product of two tensors of degrees n1 and n2, respectively.
  // The tensors are given by arrays of coefficients that correspond to lexicographically ordered non-decreasing index sequences, for example:
  //   [111, 112, 122, 222] or [11, 12, 13, 22, 23, 33]
  // In the inner product we assume that n2<n1, and that we contract n2 indices from both tensors with each other.
  // So suppose we have a tensor of the form 112 and contract it with one of the form 22, we get nothing (since 1.2=0 and regardless of how we permute the indices, we always have one 1.2 factor in there).
  assert(n1>n2, "n1 should be greater than n2")

  // Gather all non-decreasing index sequences for the various tensors involved
  var indcnt1 = getIndicesAndCounts(n1, d)
  var indcnt2 = getIndicesAndCounts(n2, d)
  var indcntr = getIndicesAndCounts(n1-n2, d)
  var inds1 = indcnt1.indices//, cnts1 = indcnt1.counts
  var inds2 = indcnt2.indices, cnts2 = indcnt2.counts
  var indsr = indcntr.indices//, cntsr = indcntr.counts

  // For each result index sequences, find the source index sequences that participate and form the appropriate inner product
  var components = [], terms
  for(var ir=0; ir<indsr.length; ir++) {
    terms = []
    for(var i2=0; i2<inds2.length; i2++) { // Every index sequence of tensor 2 participates (as it's the "smaller" tensor)
      var ind1 = indsr[ir].concat(inds2[i2]).sort(function(a,b){return a-b}) // The index sequence in tensor 1 that participates must be exactly equal to the union of the two other index sequences
      var i1 = binarySearch(ind1, inds1, compareLex)
      assert(i1 !== undefined, "Can not find desired index in tensor 1?!")
      terms.push(cnts2[i2] + "*A[" + i1 + "]*B[" + i2 + "]")
    }
    components.push(terms.join("+"))
  }
  return new Function("A", "B", "return [" + components.join(",") + "]")
}

function getIndicesAndCounts(n, d) {
  assert((typeof n) === "number" || n instanceof Number, "n must be a number")
  assert((typeof d) === "number" || d instanceof Number, "n must be a number")
  assert((n%1)===0 && (d%1)===0, "n and d must be integers")
  assert(n>0 && d>0, "n and d must be positive")

  function multUpdate(i1, i2) { // Update coef, indexpairHist and denom
    var m1 = indexpairhist[i1]
    var m2 = indexpairhist[i2]
    mult = mult*m2/(m1+1)
    indexpairhist[i1] += 1
    indexpairhist[i2] -= 1
  }

  // Gather multiplicities for all (non-decreasing) sequences of indices in lexicographic order
  var indices = new Int32Array(n), indexpairhist = new Int32Array(d), allIndices = [] , mult = 1, mults = [mult]
  indexpairhist[0] = n // Initially we have n pairs of index zero
  allIndices.push(Array.of.apply(null, indices))
  while(Math.min.apply(null, indices)<d-1) { // Loop while something can be increased
    for(var i=n; i-->0;) { // Find right-most index that can be increased
      if (indices[i]<d-1) break
    }
    multUpdate(indices[i]+1, indices[i])
    indices[i]++ // This is guaranteed to work, because the while entry condition guarantees the for loop will find some index to increase
    for(var j=i+1; j<n; j++) { // Make sure rest of the indices are equal to indices[i] (lowest non-decreasing sequence with this prefix)
      if (indices[j] !== indices[i]) multUpdate(indices[i], indices[j])
      indices[j] = indices[i]
    }
    allIndices.push(Array.of.apply(null, indices))
    mults.push(mult) // Push onto list with multiplicities
  }

  return {indices: allIndices, counts: mults}
}

// TODO: Generalize this function to higher dimensions (once that is done, the rest follows fairly easily), or make sure it is no longer needed.
// The problem is that in general it is fiendishly hard to generate a "uniform" distribution on a sphere (in fact, even defining what it means for a distribution of points to be uniform is not trivial).
// The most realistic possibility is probably generating a whole bunch of vectors randomly and then using some sort of potential minimization procedure to distribute them even more evenly.
function constructFrame(numDirs, tFV) {
  var i, a, frame = []
  for(i=0; i<numDirs; i++) {
    a = Math.PI*i/numDirs // Cover only half of the circle, as the tensors are symmetric.
    frame.push(tFV([Math.cos(a),Math.sin(a)]))
  }
  return frame
}

function constructVecs(numDirs) {
  var i, a, frame = []
  for(i=0; i<numDirs; i++) {
    a = Math.PI*i/numDirs // Cover only half of the circle, as the tensors are symmetric.
    frame.push([Math.cos(a),Math.sin(a)])
  }
  return frame
}

function newZeroArray(l) {
  var list = new Array(l)
  for(var i=0; i<l; i++) {
    list[i] = 0
  }
  return list
}

function vec_muladd(a, lambda, b) {
  var c = new Array(a.length)
  for(var i=0; i<a.length; i++) {
    c[i] = a[i] + lambda*b[i]
  }
  return c
}

function vec_muladdeq(a, lambda, b) {
  for(var i=0; i<a.length; i++) {
    a[i] += lambda*b[i]
  }
  return a
}

function vec_sub(a, b) {
  var c = new Array(a.length)
  for(var i=0; i<a.length; i++) {
    c[i] = a[i] - b[i]
  }
  return c
}

function vec_diffo_sqr(a, b) {
  var d1 = 0, d2 = 0
  for(var i=0; i<a.length; i++) {
    d1 += (a[i] - b[i])*(a[i] - b[i])
    d2 += (a[i] + b[i])*(a[i] + b[i])
  }
  return d1 <= d2 ? d1 : d2
}

function compareLex(A,B) {
  var n = A.length
  for(var i=0; i<n; i++) {
    if (A[i]<B[i]) return -1
    if (A[i]>B[i]) return 1
  }
  return 0
}

function binarySearch(t, list, comp) {
  var min = 0, max = list.length, mid
  while(min < max) {
    mid = min + ((max-min)>>1)
    if (comp(list[mid],t)<0) {
      min = mid+1
    } else {
      max = mid
    }
  }
  if (min === max && comp(list[min],t) === 0) {
    return min
  } else {
    return undefined
  }
}

if (!Array.of) {
  Array.of = function() {
    return Array.prototype.slice.call(arguments);
  };
}
