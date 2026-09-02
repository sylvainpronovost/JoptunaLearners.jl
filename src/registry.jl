const _MODEL_SPECS = Dict{Symbol,ModelSpec}()

_range(kind, values; condition=nothing) = (; kind, values, condition)

function _register!(name, public_name; tabular=false, windows=false, entity_id=false, temporal=false,
                    defaults=(;), schema=(;), context=())
    _MODEL_SPECS[name] = ModelSpec(name, public_name, tabular, windows, entity_id, temporal,
        :point, Tuple(context), defaults, schema)
end

_register!(:mlp, :MLPRegressor; tabular=true,
    defaults=(hidden_dims=(128,64), dropout=0.1),
    schema=(n_layers=_range(:int, (1,3)), hidden_dim=_range(:categorical, [64,128,256]), dropout=_range(:float, (0.0,0.4))))
_register!(:tabular_mixer, :TabularMixer; tabular=true,
    defaults=(depth=2, token_dim=32, channel_dim=64, hidden_dim=64, dropout=0.1,
              use_token=true, use_channel=true, use_norm=true, use_residual=true),
    schema=(depth=_range(:int,(1,4)), token_dim=_range(:categorical,[16,32,64]), channel_dim=_range(:categorical,[32,64,128]), hidden_dim=_range(:categorical,[32,64,128]), dropout=_range(:float,(0.0,0.4))))
_register!(:tabular_resnet, :TabularResNet; tabular=true,
    defaults=(n_blocks=3, block_width=256, hidden_factor=2.0, dropout=0.1),
    schema=(n_blocks=_range(:int,(2,8)), block_width=_range(:categorical,[64,128,256,512]), hidden_factor=_range(:float,(1.0,4.0)), dropout=_range(:float,(0.0,0.4))))
_register!(:window_mlp, :WindowMLP; windows=true, temporal=true,
    defaults=(hidden_dims=(256,128), dropout=0.1),
    schema=(n_layers=_range(:int,(1,3)), hidden_dim=_range(:categorical,[64,128,256,512]), dropout=_range(:float,(0.0,0.4))))
_register!(:window_linear, :WindowLinear; windows=true, temporal=true,
    defaults=(dropout=0.0,), schema=(dropout=_range(:float,(0.0,0.2)),))
_register!(:window_nlinear, :WindowNLinear; windows=true, temporal=true,
    defaults=(dropout=0.0,), schema=(dropout=_range(:float,(0.0,0.2)),))
_register!(:window_dlinear, :WindowDLinear; windows=true, temporal=true,
    defaults=(kernel_size=5, dropout=0.0),
    schema=(kernel_size=_range(:categorical,[3,5,9]), dropout=_range(:float,(0.0,0.2))))
_register!(:tsmixer, :TSMixer; windows=true, temporal=true,
    defaults=(depth=2, token_dim=32, channel_dim=64, hidden_dim=64, dropout=0.1, revin=true),
    schema=(depth=_range(:int,(1,4)), token_dim=_range(:categorical,[16,32,64]), channel_dim=_range(:categorical,[32,64,128]), hidden_dim=_range(:categorical,[32,64,128]), dropout=_range(:float,(0.0,0.4)), revin=_range(:categorical,[true,false])))
_register!(:film, :FiLMTSMixer; windows=true, entity_id=true, temporal=true, context=(:entity_vocabulary,),
    defaults=(emb_dim=8, depth=2, token_dim=32, channel_dim=64, hidden_dim=64, dropout=0.1, revin=true),
    schema=(emb_dim=_range(:categorical,[4,8,16]), depth=_range(:int,(1,4)), token_dim=_range(:categorical,[16,32,64]), channel_dim=_range(:categorical,[32,64,128]), hidden_dim=_range(:categorical,[32,64,128]), dropout=_range(:float,(0.0,0.4)), revin=_range(:categorical,[true,false])))
for (name, public) in ((:tcn,:TCN), (:tcn_v2,:TCNV2))
    _register!(name, public; windows=true, temporal=true,
        defaults=(hidden_dim=64, depth=3, kernel_size=3, dropout=0.1, auto_depth=true, use_weight_norm=true, revin=true),
        schema=(hidden_dim=_range(:categorical,[32,64,128]), depth=_range(:int,(2,5)), kernel_size=_range(:categorical,[2,3,5]), dropout=_range(:float,(0.0,0.4)), revin=_range(:categorical,[true,false])))
end
_register!(:patchtst, :PatchTSTLite; windows=true, temporal=true,
    defaults=(patch_len=8, stride=4, hidden_dim=64, depth=2, n_heads=4, dropout=0.1),
    schema=(patch_len=_range(:categorical,[4,8,12,16]), stride=_range(:categorical,[2,4,8]), hidden_dim=_range(:categorical,[32,64,128]), n_heads=_range(:categorical,[2,4,8]), depth=_range(:int,(1,4)), dropout=_range(:float,(0.0,0.4))))
_register!(:tft, :TFTLite; windows=true, temporal=true,
    defaults=(hidden_dim=64, recurrent_layers=1, n_heads=4, dropout=0.1),
    schema=(hidden_dim=_range(:categorical,[32,64,128]), n_heads=_range(:categorical,[2,4,8]), recurrent_layers=_range(:int,(1,2)), dropout=_range(:float,(0.0,0.4))))
_register!(:mamba, :MambaLite; windows=true, temporal=true,
    defaults=(hidden_dim=64, depth=2, kernel_size=3, dropout=0.1),
    schema=(hidden_dim=_range(:categorical,[32,64,128]), depth=_range(:int,(1,4)), kernel_size=_range(:categorical,[2,3,5]), dropout=_range(:float,(0.0,0.4))))

model_specs() = sort!(collect(values(_MODEL_SPECS)); by=x -> String(x.name))

function model_spec(name::Union{Symbol,AbstractString})
    key = Symbol(name)
    haskey(_MODEL_SPECS, key) || throw(ArgumentError("unsupported model: $name"))
    _MODEL_SPECS[key]
end

function model_config(name::Union{Symbol,AbstractString}; kwargs...)
    spec = model_spec(name)
    merge(spec.defaults, (; kwargs...))
end

hyperparameter_schema(spec::ModelSpec) = spec.schema
hyperparameter_schema(name::Union{Symbol,AbstractString}) = hyperparameter_schema(model_spec(name))
