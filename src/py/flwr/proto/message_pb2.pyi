"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import flwr.proto.error_pb2
import flwr.proto.recorddict_pb2
import flwr.proto.transport_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Message(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    METADATA_FIELD_NUMBER: builtins.int
    CONTENT_FIELD_NUMBER: builtins.int
    ERROR_FIELD_NUMBER: builtins.int
    @property
    def metadata(self) -> global___Metadata: ...
    @property
    def content(self) -> flwr.proto.recorddict_pb2.RecordDict: ...
    @property
    def error(self) -> flwr.proto.error_pb2.Error: ...
    def __init__(self,
        *,
        metadata: typing.Optional[global___Metadata] = ...,
        content: typing.Optional[flwr.proto.recorddict_pb2.RecordDict] = ...,
        error: typing.Optional[flwr.proto.error_pb2.Error] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["content",b"content","error",b"error","metadata",b"metadata"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["content",b"content","error",b"error","metadata",b"metadata"]) -> None: ...
global___Message = Message

class Context(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class NodeConfigEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> flwr.proto.transport_pb2.Scalar: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[flwr.proto.transport_pb2.Scalar] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    class RunConfigEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> flwr.proto.transport_pb2.Scalar: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[flwr.proto.transport_pb2.Scalar] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    RUN_ID_FIELD_NUMBER: builtins.int
    NODE_ID_FIELD_NUMBER: builtins.int
    NODE_CONFIG_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    RUN_CONFIG_FIELD_NUMBER: builtins.int
    run_id: builtins.int
    node_id: builtins.int
    @property
    def node_config(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, flwr.proto.transport_pb2.Scalar]: ...
    @property
    def state(self) -> flwr.proto.recorddict_pb2.RecordDict: ...
    @property
    def run_config(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, flwr.proto.transport_pb2.Scalar]: ...
    def __init__(self,
        *,
        run_id: builtins.int = ...,
        node_id: builtins.int = ...,
        node_config: typing.Optional[typing.Mapping[typing.Text, flwr.proto.transport_pb2.Scalar]] = ...,
        state: typing.Optional[flwr.proto.recorddict_pb2.RecordDict] = ...,
        run_config: typing.Optional[typing.Mapping[typing.Text, flwr.proto.transport_pb2.Scalar]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["state",b"state"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["node_config",b"node_config","node_id",b"node_id","run_config",b"run_config","run_id",b"run_id","state",b"state"]) -> None: ...
global___Context = Context

class Metadata(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    RUN_ID_FIELD_NUMBER: builtins.int
    MESSAGE_ID_FIELD_NUMBER: builtins.int
    SRC_NODE_ID_FIELD_NUMBER: builtins.int
    DST_NODE_ID_FIELD_NUMBER: builtins.int
    REPLY_TO_MESSAGE_ID_FIELD_NUMBER: builtins.int
    GROUP_ID_FIELD_NUMBER: builtins.int
    TTL_FIELD_NUMBER: builtins.int
    MESSAGE_TYPE_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    run_id: builtins.int
    message_id: typing.Text
    src_node_id: builtins.int
    dst_node_id: builtins.int
    reply_to_message_id: typing.Text
    group_id: typing.Text
    ttl: builtins.float
    message_type: typing.Text
    created_at: builtins.float
    def __init__(self,
        *,
        run_id: builtins.int = ...,
        message_id: typing.Text = ...,
        src_node_id: builtins.int = ...,
        dst_node_id: builtins.int = ...,
        reply_to_message_id: typing.Text = ...,
        group_id: typing.Text = ...,
        ttl: builtins.float = ...,
        message_type: typing.Text = ...,
        created_at: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["created_at",b"created_at","dst_node_id",b"dst_node_id","group_id",b"group_id","message_id",b"message_id","message_type",b"message_type","reply_to_message_id",b"reply_to_message_id","run_id",b"run_id","src_node_id",b"src_node_id","ttl",b"ttl"]) -> None: ...
global___Metadata = Metadata
