import dlib
from dlib import rectangle
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
import configparser

# models.py
class Model:
    __metaclass__ = ABCMeta

    nParams = 0

    #zwraca wektor rezyduow przy danych parametrach modelu, wektorze wejsciowym i oczekiwanych wektorze wyjsciowym
    def residual(self, params, x, y):
        r = y - self.fun(x, params)
        r = r.flatten()

        return r

    #zwraca wartosci zwracane przez model przy danych parametrach i wektorze wejsciowym
    @abstractmethod
    def fun(self, x, params):
        pass

    #zwraca jakobian
    @abstractmethod
    def jacobian(self, params, x, y):
        pass

    #zwraca zbior przykladowych parametrow modelu
    @abstractmethod
    def getExampleParameters(self):
        pass

    #zwraca inny zbior przykladowych parametrow
    @abstractmethod
    def getInitialParameters(self):
        pass

class OrthographicProjectionBlendshapes(Model):
    nParams = 6

    def __init__(self, nBlendshapes):
        self.nBlendshapes = nBlendshapes
        self.nParams += nBlendshapes

    def fun(self, x, params):
        #skalowanie
        s = params[0]
        #rotacja
        r = params[1:4]
        #przesuniecie (translacja)
        t = params[4:6]
        w = params[6:]

        mean3DShape = x[0]
        blendshapes = x[1]

        #macierz rotacji z wektora rotacji, wzor Rodriguesa
        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

        projected = s * np.dot(P, shape3D) + t[:, np.newaxis]

        return projected

    def jacobian(self, params, x, y):
        s = params[0]
        r = params[1:4]
        t = params[4:6]
        w = params[6:]

        mean3DShape = x[0]
        blendshapes = x[1]

        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

        nPoints = mean3DShape.shape[1]
        
        #nSamples * 2 poniewaz kazdy punkt ma dwa wymiary (x i y)
        jacobian = np.zeros((nPoints * 2, self.nParams))

        jacobian[:, 0] = np.dot(P, shape3D).flatten()

        stepSize = 10e-4
        step = np.zeros(self.nParams)
        step[1] = stepSize;
        jacobian[:, 1] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()
        step = np.zeros(self.nParams)
        step[2] = stepSize;
        jacobian[:, 2] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()
        step = np.zeros(self.nParams)
        step[3] = stepSize;
        jacobian[:, 3] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()

        jacobian[:nPoints, 4] = 1
        jacobian[nPoints:, 5] = 1

        startIdx = self.nParams - self.nBlendshapes
        for i in range(self.nBlendshapes):
            jacobian[:, i + startIdx] = s * np.dot(P, blendshapes[i]).flatten()

        return jacobian

    #nie uzywane
    def getExampleParameters(self):
        params = np.zeros(self.nParams)
        params[0] = 1

        return params
    
    def getInitialParameters(self, x, y):
        mean3DShape = x.T
        shape2D = y.T
   
        shape3DCentered = mean3DShape - np.mean(mean3DShape, axis=0)
        shape2DCentered = shape2D - np.mean(shape2D, axis=0)

        scale = np.linalg.norm(shape2DCentered) / np.linalg.norm(shape3DCentered[:, :2]) 
        t = np.mean(shape2D, axis=0) - np.mean(mean3DShape[:, :2], axis=0)

        params = np.zeros(self.nParams)
        params[0] = scale
        params[4] = t[0]
        params[5] = t[1]

        return params

# NonLinearLeastSquares.py
def LineSearchFun(alpha, x, d, fun, args):
    r = fun(x + alpha * d, *args)
    return np.sum(r**2)

def GaussNewton(x0, residual, jacobian, args, maxIter=10, eps=10e-7, verbose=1):
    x = np.array(x0, dtype=np.float64)

    oldCost = -1
    for i in range(maxIter):
        r = residual(x, *args) # residual
        cost = np.sum(r**2)

        if verbose > 0:
            print ("Cost at iteration " + str(i) + ": " + str(cost))

        #print( "cost:", cost, "i:", i )

        if (cost < eps or abs(cost - oldCost) < eps):
            break
        oldCost = cost

        J = jacobian(x, *args) #jacobian
        grad = np.dot(J.T, r)
        H = np.dot(J.T, J)
        direction = np.linalg.solve(H, grad)

        #optymalizacja dlugosci kroku
        #lineSearchRes = optimize.minimize_scalar(LineSearchFun, args=(x, direction, residual, args))
        #dlugosc kroku
        #alpha = lineSearchRes["x"]
        #x = x + alpha * direction
        x = x + direction
        
    if verbose > 0:
        print ("Gauss Netwon finished after "  + str(i + 1) + " iterations")
        r = residual(x, *args)
        cost = np.sum(r**2)
        print ("cost = " + str(cost))
        print ("x = " + str(x))

    return x
    
# ImageProcessing.py
def blendImages(src, dst, mask, featherAmount=0.2):
    #indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    #te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    return composedImg

#uwaga, tutaj src to obraz, z ktorego brany bedzie kolor
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    #indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    #src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst
    
# drawing.py
def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)

def drawCross(img, params, center=(100, 100), scale=30.0):
    R = cv2.Rodrigues(params[1:4])[0]

    points = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points = np.dot(points, R.T)
    points2D = points[:, :2]

    points2D = (points2D * scale + center).astype(np.int32)
    
    cv2.line(img, (center[0], center[1]), (points2D[0, 0], points2D[0, 1]), (255, 0, 0), 3)
    cv2.line(img, (center[0], center[1]), (points2D[1, 0], points2D[1, 1]), (0, 255, 0), 3)
    cv2.line(img, (center[0], center[1]), (points2D[2, 0], points2D[2, 1]), (0, 0, 255), 3)

def drawMesh(img, shape, mesh, color=(255, 0, 0)):
    for triangle in mesh:
        point1 = shape[triangle[0]].astype(np.int32)
        point2 = shape[triangle[1]].astype(np.int32)
        point3 = shape[triangle[2]].astype(np.int32)

        cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), (255, 0, 0), 1)
        cv2.line(img, (point2[0], point2[1]), (point3[0], point3[1]), (255, 0, 0), 1)
        cv2.line(img, (point3[0], point3[1]), (point1[0], point1[1]), (255, 0, 0), 1)

def drawProjectedShape(img, x, projection, mesh, params, lockedTranslation=False):
    localParams = np.copy(params)

    if lockedTranslation:
        localParams[4] = 100
        localParams[5] = 200

    projectedShape = projection.fun(x, localParams)

    drawPoints(img, projectedShape.T, (0, 0, 255))
    drawMesh(img, projectedShape.T, mesh)
    drawCross(img, params)

# utils.py
def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ

def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]

def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)

    return mesh

def getShape3D(mean3DShape, blendshapes, params):
    #skalowanie
    s = params[0]
    #rotacja
    r = params[1:4]
    #przesuniecie (translacja)
    t = params[4:6]
    w = params[6:]

    #macierz rotacji z wektora rotacji, wzor Rodriguesa
    R = cv2.Rodrigues(r)[0]
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D

def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D

def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))


    #detekcja twarzy
    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))

        #detekcja punktow charakterystycznych twarzy
        dlibShape = predictor(img, faceRectangle)
        
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        #transpozycja, zeby ksztalt byl 2 x n a nie n x 2, pozniej ulatwia to obliczenia
        shape2D = shape2D.T

        shapes2D.append(shape2D)

    return shapes2D
    

def getFaceTextureCoords(projectionModel, img, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor):
    #projectionModel = OrthographicProjectionBlendshapes(blendshapes.shape[0])

    keypoints = getFaceKeypoints(img, detector, predictor)[0]
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)

    return textureCoords

def pymain():
    print ("Press T to draw the keypoints and the 3D model")
    print ("Press R to start recording to a video file")

    config = configparser.ConfigParser()
    config.read('data.ini')
    predictor_path = config['DEFAULT']['predictor']
    image_name = config['DEFAULT']['image']
    model_path = config['DEFAULT']['model']
    
    maxImageSizeForDetection = 320

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    mean3DShape, blendshapes, mesh, idxs3D, idxs2D = load3DFaceModel(model_path)

    projectionModel = OrthographicProjectionBlendshapes(blendshapes.shape[0])

    modelParams = None
    lockedTranslation = False
    drawOverlay = False
    writer = None

    textureImg = cv2.imread(image_name)
    textureCoords = getFaceTextureCoords(projectionModel,textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
    slice_mean3DShape = mean3DShape[:, idxs3D]
    slice_blendshapes = blendshapes[:, :, idxs3D]
    
    header = str(mean3DShape.shape)
    np.savetxt("mean3DShape", mean3DShape.flatten(), fmt='%.4e', header=header)
    header = str(mesh.shape)
    np.savetxt("mesh", mesh.flatten(), fmt='%i', header=header)
    header = str(idxs3D.shape)
    np.savetxt("idxs3D", idxs3D.flatten(), fmt='%i', header=header)
    header = str(idxs2D.shape)
    np.savetxt("idxs2D", idxs2D.flatten(), fmt='%i', header=header)
    header = str(blendshapes.shape)
    np.savetxt("blendshapes", blendshapes.flatten(), fmt='%.4e', header=header)
    header = str(textureCoords.shape)
    np.savetxt("textureCoords", textureCoords.flatten(), fmt='%.4e', header=header)
    header = str(slice_mean3DShape.shape)
    np.savetxt("slice_mean3DShape", slice_mean3DShape.flatten(), fmt='%.4e', header=header)
    header = str(slice_blendshapes.shape)
    np.savetxt("slice_blendshapes", slice_blendshapes.flatten(), fmt='%.4e', header=header)
    
    import ctypes
    ctypes.windll.user32.MessageBoxW(0, "Finshed!", "Dump file", 0)

if __name__ == '__main__':
    pymain()