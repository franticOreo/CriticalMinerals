
def get_mineral_estimate(minedex_site_date): #main function 
    '''minedex_site_date: row from minedex dataset that is filtered by site and a date

    # get usable model inputs (what minerals are already measure in MINEDEX)
    # train multiple models with usable inputs
    # finding all assays with inputs + target critical mineral
    # train all models
    # benchmark train/test scores
    # choose best model
    # get most accurate prediction of critical mineral

    returns: cm_t: critical mineral in tonnes 
    '''

    # get commodities that occur in mindex data, that can be 
    # used for prediction
    # Minedex data for sites varies in amounts of commodities
    # so... some models will have two inputs, some more (6 inputs)
    list_of_inputs = get_commodity_inputs(mindex_site_date)

    # extract commodity values for use for prediction
    commodity_values = get_commodity_values(mindex_site_date)

    # get assays containing relevant commodities
    filtered_assays = filter_assays(assay_df, list_of_inputs)

    # if there is a trained model with the existing group of commodities
    # use the save model and predict with it.
    trained_model = search_for_trained_model(list_of_inputs)
    
    if trained_model:
        # prediciton for critical mineral in ppm.  
        cm_pred = trained_model.predict(commodity_values)

    # if no relevant trained model
    # benchmark a range of models
    # e.g Linear Regression vs Random Forrest.
    # pick most accurate according to train/test scores

    cm_pred = train_new_models(filtered_assays, commodity_values)

    cm_tonnes = ppm_to_tonnes(cm_pred)

    return  cm_tonnes

def filter_assays(assay_df, list_of_inputs):
    # filter assay dataframe by inputs
    return filtered_assays

def search_for_trained_model(list_of_inputs):
    # somehow search for trained models in directory
    # if model has already been trained with the same inputs
    # return model
    return trained_model

def train_new_models(filtered_assays, commodity_values):
    # train a range of models
    for model in models:
        X_train, X_test, y_train, y_test = split_data(filtered_assays)
        model.fit(filtered_assays)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

    # decide which train/test preds are the most accurate
    best_model = pick_best_model()

    best_model_preds = best_model.predict(commodity_values)
    # return most accurate models preds for filtered assays
    return best_model_preds

def get_commodity_inputs(mindex_site_date):
    # get unique commodities from mindex_site_date
    return list_of_inputs

def ppm_to_tonnes(cm, assay):
    # somehow convert concentration to tonnes
    return cm_tonnes


