var decompose = require("./decompose.js")

console.log(decompose.makeTensorDot(1,2).toString())
console.log(decompose.makeTensorDot(2,2).toString())
console.log(decompose.makeTensorDot(3,2).toString())
console.log(decompose.makeTensorDot(3,3).toString())

console.log(decompose.makePartialInnerProduct(2, 1, 2).toString())
console.log(decompose.makePartialInnerProduct(4, 3, 2).toString())
console.log(decompose.makePartialInnerProduct(4, 2, 2).toString())
console.log(decompose.makePartialInnerProduct(4, 2, 3).toString())

console.log(decompose.constructIdentityTensor(2,2))
console.log(decompose.constructIdentityTensor(2,3))
console.log(decompose.constructIdentityTensor(4,2))
console.log(decompose.constructIdentityTensor(4,3))
console.log(decompose.constructIdentityTensor(6,3))

console.log(decompose.makeTensorFromVector(1,2).toString())
console.log(decompose.makeTensorFromVector(2,2).toString())
console.log(decompose.makeTensorFromVector(3,2).toString())
console.log(decompose.makeTensorFromVector(4,2).toString())
console.log(decompose.makeTensorFromVector(3,3).toString())

console.log("Easy")
console.log(decompose([1,0,0,0,0]))

console.log("Easy 8")
console.log(decompose([1,0,0,0,0,0,0,0,0]))

// Original vector: [0.92106099400288508280, 0.38941834230865049167]
console.log("Easy (rotated)")
console.log(decompose([0.71970341438592161968, 0.30428572310506883595, 0.12864994028766109078, 0.054392322344692544864, 0.022996705038756198764]))

console.log("Easy (orthogonal)")
console.log(decompose([0,0,0,0,1]))

console.log("Pair")
console.log(decompose([2, 2, 4, 8, 16])) // [1, 2, 4, 8, 16] + [1,0,0,0,0]

console.log("Easy pair 8")
console.log(decompose([1,0,0,0,0,0,0,0,1]))

console.log("Easy + identity tensor")
console.log(decompose([2, 0, 0.33333333333333333333, 0, 1]))

console.log("Easy (rotated) + identity tensor")
console.log(decompose([1.71970341438592161968, 0.30428572310506883595, 0.12864994028766109078+0.33333333333333333333, 0.054392322344692544864, 1.022996705038756198764]))

console.log("Pair + identity tensor")
console.log(decompose([3, 2, 4.33333333333333333333, 8, 17]))

// This one works reasonably well, but takes a /long/ time to converge to sufficient precision.
// And interestingly, once it converges it finds a solution that is slightly different from the original input,
// yet yields the same tensor... The original tensor was constructed as: [1,0]^4 + 3*[1,2]^4 + 2*[2,1]^4 (note that \|[1,2]\|^4 = 25)
console.log("Trio")
console.log(decompose([36, 22, 20, 28, 50]))

// The following are expected to give trouble
console.log("Pair + noise")
console.log(decompose([2.37917, 1.95782, 4.54709, 8.53602, 15.8799]))

console.log("Pair with one negative component")
console.log(decompose([0, 2, 4, 8, 16]))

console.log("Trio + noise")
console.log(decompose([36.0969, 22.2575, 20.2784, 27.869, 48.9781]))

// This is a tricky one, as the negative component is larger than the positive one
console.log("Pair with large negative component")
console.log(decompose([-40, 2, 4, 8, 16]))
