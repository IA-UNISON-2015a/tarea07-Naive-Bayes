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
        y valos un diccionario donde
            vals[var] = set([val1, ..., valm])
        son los valores que puede tomar la variable var. Si vals es None, entonces vals se construye
        con los valores que se tienen en el conjunto de aprendizaje.

        clases es un conjuto (o lista) [clase1, ..., clasek], con el nombre de las k clases que nos
        interesan. Si clases == None o vals == None entonces no se inicializa el diccionario de
        frecuencias, y se espera a que inicie el aparendizaje para inicializar las cuentas. Para
        mas información sobre como manejaremos las frecuencias, revisar la documentación del método
        genera_cuentas.

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

        """
        if self.vals is None:
            self.vals = {self.vars[i]: set([datos[j][i] for j in range(len(datos))]) for i in range(len(self.vars))}
        if self.cls_nombre is None:
            self.cls_nombre = set(clases)
        if self.cuentas is None:
            self.genera_cuentas()

        # Aqui comienza el código para hacer el llenado de las CPT
        #print datos
        # Primero se llena el valor de las cuentas para calcular la probabilidad a priori
        tam=len(clases)
            
        for class1 in self.cls_nombre:
            #---------------------------------------------------
            # agregar aqui el código
            for x in range (tam):
                if clases[x]== class1:
                    #print 'entra: ', clases[x]
                    self.cuentas['clases'][class1] += 1
            #---------------------------------------------------
           
        # Ahora se llena el valor de las cuentas por cada atributo y para cada posible clase
        for i in range(len(self.vars)):
            print 'Primer cilco: ',i
            for clase in self.cls_nombre:

                #--------------------------------------------------
                # agregar aquí el código
                # self.vars, variables que se guardan en i
                #Clase, lo otro
                for j in self.cuentas[self.vars[i]][clase]:
                    #print 'J: ',j, ',Vars: ',self.vars[i]
                    for k in range(len(datos)):
                        if datos[k][i] == j and clases[k] == clase:
                            self.cuentas[self.vars[i]][clase][j] += 1
                #pass
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
        print 'Len: ',len(datos), 'datos: ',datos
        #print 'Vals: ', self.vals
        #print 'len Vals: ', len(self.vals)
        #print 'Cuentas: ', self.cuentas
        #print 'Nombre: ', self.cls_nombre



        #---------------------------------------------------
        # agregar aquí el código
        #---------------------------------------------------

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
    if clase_test == ['P', 'N']:

        print "La tercera prueba se completó con exito"

if __name__ == "__main__":
    test()