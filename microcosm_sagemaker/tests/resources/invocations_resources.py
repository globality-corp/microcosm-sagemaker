from marshmallow import Schema, fields
from microcosm_flask.paging import PageSchema


class ClassificationResultSchema(Schema):
    uri = fields.Url(
        required=True,
        relative=False,
    )
    score = fields.Float(
        required=True
    )


class NewPredictionSchema(PageSchema):
    simpleArg = fields.Float(
        required=True,
        attribute="simple_arg",
    )
