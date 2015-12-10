#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nb.py
------------

Clase genérica para realizar el método de clasificación de naive bayes.

"""

__author__ = 'juliowaissman'


class NaiveBayes(object):
    """
    Clase genérica del clasificador naive bayes, para entradas con dominio discreto y finito

    """

    def __init__(self, variables=(), vals=None, cls_nombre=None):
        """
        Inicializa el algoritmo con vars el nombre de las variables de los atributos (no la clase),
        y valos un diccionario donde vals[var] = set([val1, ..., valm])
        son los valores que puede tomar la variable var. Si vals es None, entonces vals se construye
        con los valores que se tienen en el conjunto de aprendizaje.

        clases es un conjuto (o lista) [clase1, ..., clasek], con el nombre de las k clases que nos
        interesan. Si clases == None o vals == None entonces no se inicializa el diccionario de
        frecuencias, y se espera a que inicie el aparendizaje para inicializar las cuentas. Para
        mas información sobre como manejaremos las frecuencias, revisar la documentación del método
        genera_cuentas.

        vars = nombre de variables de atributos
        vals = valores que puede tomar
        clases = [clase1, clasek] nombre de las k clases en las que se puede clasificar.

        """
        self.vars = variables
        self.vals = vals
        self.cls_nombre = cls_nombre
        self.num_datos = 0
        if vals is not None and cls_nombre is not None:
            self.genera_cuentas()
        else:
            self.cuentas = None

    def genera_cuentas(self):
        """
        Para poder hacer un aprendizaje incremental, esto es, poder agregar nuevos ejemplos de aprendizaje en
        linea, en lugar de guardar las CPTs, vamos a utilizar un diccionario con frecuencias.

        self.frecuencias['clases'] es un diccionario, donde en cada nombre de clase guarda la frecuencia encontrada
        (esto permite inclusive agregar nuevas clases en linea). Así, por cada valor en la lista self.cls_nombre
        habra una entrada del diccionario self.frecuencias['clases].

        Por cada variable habrá otros dicionarios, de manera que
            self.frecuencias[var][clase][val]
        es el numero de ocurrencias del valor val, de la variable var, cuando los datos están asociados a la
        clase clase.

        """
        self.cuentas = {var: {clase: {val: 0 for val in self.vals[var]}
                              for clase in self.cls_nombre}
                        for var in self.vars}
        self.cuentas['clases'] = {clase: 0 for clase in self.cls_nombre}

    def aprende(self, datos, clases):
        """
        Aprende los valores de la CPT, es el trabajo a realizar

        datos =[datum1,datum2,..datumN]
        datum1 = [datum1a, datum1b, ...]
        clase = [clase1,...claseN]
        claseN = clasificacion asociada a datum1--N

        """
        if self.vals is None:
            self.vals = {self.vars[i]: set([datos[j][i] for j in range(len(datos))]) for i in range(len(self.vars))}
        if self.cls_nombre is None:
            self.cls_nombre = set(clases)
        if self.cuentas is None:
            self.genera_cuentas()

        # Aqui comienza el código para hacer el llenado de las CPT

        # Primero se llena el valor de las cuentas para calcular la probabilidad a priori
        for clase in self.cls_nombre: #Para cada una de las clases definidas en nuestro clasificador
            #---------------------------------------------------
            # agregar aqui el código
            for i in range(len(clases)):
                if clases[i] == clase: #Encontramos el match para cada clase recorriendo el arreglo
                    self.cuentas['clases'][clase] += 1 #Le damos un +1 a la clases con match con esto podremos calcular frecuencias.

                    #Incidencia de una clase en total.
            #---------------------------------------------------

        # Ahora se llena el valor de las cuentas por cada atributo y para cada posible clase
        for i in range(len(self.vars)): #Por cada variable de atributo
            for clase in self.cls_nombre:  #Por cada clase existente

                #--------------------------------------------------
                # agregar aquí el código
                for j in self.cuentas[self.vars[i]][clase]: #Por el numero de variables
                    for k in range(len(datos)):
                        if datos[k][i] == j and clases[k] == clase:
                            self.cuentas[self.vars[i]][clase][j] +=1

                            #Incidencia de una clase con una val en particular.
                #--------------------------------------------------

    def reconoce(self, datos):
        """
        Identifica la clase a la que pertenece cada uno de los datos que se solicite, de acuerdo al clasificador
        datos = [dato(1), dato(2), ...] es una lista de datos para clasificar, donde
        dato(i) = [dato(i,1), ..., dato(i,n)] es el valor del dato en cada atributo. Hay que recordar que
        se puede utilizar el método de naive bayes si no se conocen todos los atributos. Si un atributo no se conoce,
        entonces dato(i,k) == None.

        Para esto vamos a utilizar el clasificador de Naive Bayes con el ajuste Laplaciano.

        """
        clases = []

        #---------------------------------------------------
        # agregar aquí el código
        #---------------------------------------------------
        base = -1;
        for clase in self.cls_nombre: # Por cada en la que se puede clasificar
            num_datos = self.cuentas['clases'][clase]
            for i in range(len(datos)): #Por cada uno de los datos de entrada
                for clase in self.cls_nombre:
                    actual = 1.0
                    for j in range(len(self.vars)):
                        actual *= self.cuentas['clases'][clase] / num_datos


                    if (actual>base):
                        c_mejor = clase
                        base = actual
                clases.append(c_mejor)


        return clases


def test():
    """
    Esta funcion sirve para poder ir probando y corrigiendo el programa.

    Hay 3 pruebas básicas, una para probar la inicialización, otra para probar el aprendizaje (o el llenado de cuentas)
    y la 3ra para probar si el reconocimiento se hace correctamente. hasta que pasen todas las pruebas no hay que pasar
    al problema que se encuentra en el archivo naive_bayes.py

    """

    variables = ('uno', 'dos')
    valores = dict(uno={1, 2, 3, 4}, dos={10, 20})
    id_clases = {'N', 'P'}
    nb = NaiveBayes(variables, valores, id_clases)

    assert nb.cuentas['clases'] == {'N': 0, 'P': 0}
    assert nb.cuentas['uno']['N'] == {1: 0, 2: 0, 3: 0, 4: 0}

    print "La primera prueba se completo con exito"

    data = [[1, 10], [2, 10], [3, 10], [4, 10], [1, 20], [2, 20], [3, 20], [4, 20]]
    clases = ['N',     'P',    'P',      'N',      'N',     'P',    'N',     'N']

    nb = NaiveBayes(('uno', 'dos'))

    assert nb.cuentas is None

    nb.aprende(data, clases)
    assert nb.cuentas['clases'] == {'N': 5, 'P': 3}
    assert nb.cuentas['uno']['N'] == {1: 2, 2: 0, 3: 1, 4: 2}
    assert nb.cuentas['dos']['P'] == {10: 2, 20: 1}
    assert nb.cuentas['dos']['N'] == {10: 2, 20: 3}

    print "La segunda prueba se completó con exito"

    data_test = [[2, 20], [4, 10]]
    clase_test = nb.reconoce(data_test)
    assert clase_test == ['P', 'N']

    print "La tercera prueba se completó con exito"

if __name__ == "__main__":
    test()