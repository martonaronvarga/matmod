pub struct DdmModel {
    pub data: Vec<Trial>,
    pub prior: DdmPrior,
}
pub struct HmmModel {
    pub sequences: Vec<Vec<Obs>>,
    pub n_states: usize,
}
