# -*- coding: utf-8 -*- 

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__)))+"/lib")

from flask import request
from flask_restx import Api, Resource
from Infer import Infer

class apiControll:
    def __init__(self,api:Api,infer:Infer):
        @api.route('/inferweb')
        class inferweb(Resource):

            def post(self):

                result_str, result_history = infer.text_gen(
                    data=request.json, type="infer")

                return {
                    "result": result_str,
                    "history": result_history
                }
        @api.route('/chatweb')
        class chatweb(Resource):

            def post(self):

                result_str, result_history = infer.text_gen(
                    data=request.json, type="chat")

                return {
                    "result": result_str,
                    "history": result_history
                }
