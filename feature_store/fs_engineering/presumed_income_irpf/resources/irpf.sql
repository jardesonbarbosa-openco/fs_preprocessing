with irpf as (
	select 
			cast(json_extract(value,'$[personId]') as INTEGER) person_id,
			id as irpf_id,
			created_at as created_at_irpf,
			date_format(created_at, '%Y%m') safra_created,
			cast(json_extract(value,'$[rev]') as INTEGER) rev,
			value
	from ml_di_datalake.irpf_person_info_history_ori
), personal_loan as (
	select
			id as loan_id,
			person_id,
			created_at,
			date_format(created_at, '%Y%m') safra_created,
			product_code,
			state,
			cast(json_extract(bank_info,'$[bankCode]') as VARCHAR) bank_code_pl,
			cast(json_extract(bank_info,'$[branchNumber]') as VARCHAR) branch_number_pl
	from ml_di_datalake.personal_loan_ori
), merger as (
	select
			irpf.person_id,
			pl.loan_id,
			irpf.irpf_id,
			irpf.created_at_irpf as time_stamp,
			date(irpf.created_at_irpf) as created_at_irpf,
			date(pl.created_at) as created_at_loan,
			irpf.safra_created,
			pl.product_code,
			pl.state,
			pl.bank_code_pl,
			pl.branch_number_pl,
			irpf.rev,
			irpf.value
	from irpf
		left join personal_loan as pl
		on irpf.person_id = pl.person_id
		and date(irpf.created_at_irpf) = date(pl.created_at)
), order_table as (
	select
		        *
		from (
		        select *, row_number() over (partition by person_id, created_at_irpf order by rev desc) as ordem
		        from merger
		     ) T1
		where
			1=1
		    and ordem = 1
	) select
			person_id,
			loan_id,
			irpf_id,
			time_stamp,
			bank_code_pl,
			branch_number_pl,
			rev,
			value
	from order_table